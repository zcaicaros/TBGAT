import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, global_mean_pool, GINConv
from torch_geometric.utils import add_self_loops, sort_edge_index
from torch_geometric.data.batch import Batch
from torch.nn.utils.rnn import pad_sequence
from parameters import args


class DGHANlayer(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl, dropout, concat, heads=1):
        super(DGHANlayer, self).__init__()
        self.dropout = dropout
        self.opsgrp_conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat)
        self.mchgrp_conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat)

    def forward(self, node_h, edge_index_pc, edge_index_mc):
        node_h_pc = F.elu(self.opsgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_pc))
        node_h_mc = F.elu(self.mchgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_mc))
        node_h = torch.mean(torch.stack([node_h_pc, node_h_mc]), dim=0, keepdim=False)
        return node_h


class DGHAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, layer_dghan=4, heads=2):
        super(DGHAN, self).__init__()
        self.layer_dghan = layer_dghan
        self.hidden_dim = hidden_dim

        ## DGHAN conv layers
        self.DGHAN_layers = torch.nn.ModuleList()

        # init DGHAN layer
        if layer_dghan == 1:
            # only DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=False, heads=heads))
        else:
            # first DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=True, heads=heads))
            # following DGHAN layers
            for layer in range(layer_dghan - 2):
                self.DGHAN_layers.append(DGHANlayer(heads * hidden_dim, hidden_dim, dropout, concat=True, heads=heads))
            # last DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(heads * hidden_dim, hidden_dim, dropout, concat=False, heads=1))

    def forward(self, x, edge_index_pc, edge_index_mc, num_instances):

        # initial layer forward
        h_node = self.DGHAN_layers[0](x, edge_index_pc, edge_index_mc)
        for layer in range(1, self.layer_dghan):
            h_node = self.DGHAN_layers[layer](h_node, edge_index_pc, edge_index_mc)

        return h_node, torch.mean(h_node.reshape(num_instances, -1, self.hidden_dim), dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_gin=4):
        super(GIN, self).__init__()
        self.layer_gin = layer_gin

        ## GIN conv layers
        self.GIN_layers = torch.nn.ModuleList()

        # init gin layer
        self.GIN_layers.append(
            GINConv(
                Sequential(Linear(in_dim, hidden_dim),
                           torch.nn.BatchNorm1d(hidden_dim),
                           ReLU(),
                           Linear(hidden_dim, hidden_dim)),
                eps=0,
                train_eps=False,
                aggr='mean',
                flow="source_to_target")
        )

        # rest gin layers
        for layer in range(layer_gin - 1):
            self.GIN_layers.append(
                GINConv(
                    Sequential(Linear(hidden_dim, hidden_dim),
                               torch.nn.BatchNorm1d(hidden_dim),
                               ReLU(),
                               Linear(hidden_dim, hidden_dim)),
                    eps=0,
                    train_eps=False,
                    aggr='mean',
                    flow="source_to_target")
            )

    def forward(self, x, edge_index, batch):

        hidden_rep = []
        node_pool_over_layer = 0
        # initial layer forward
        h = self.GIN_layers[0](x, edge_index)
        node_pool_over_layer += h
        hidden_rep.append(h)
        # rest layers forward
        for layer in range(1, self.layer_gin):
            h = self.GIN_layers[layer](h, edge_index)
            node_pool_over_layer += h
            hidden_rep.append(h)

        # Graph pool
        gPool_over_layer = 0
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += g_pool

        return node_pool_over_layer, gPool_over_layer


class TPMCAM(torch.nn.Module):
    def __init__(self,
                 in_dim=3,
                 hidden_dim=128,
                 embedding_l=4,
                 heads=1,
                 dropout=0):
        super(TPMCAM, self).__init__()

        self.embedding_gin = GIN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            layer_gin=embedding_l
        )
        self.embedding_dghan = DGHAN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layer_dghan=embedding_l,
            heads=heads
        )

    def forward(self, pyg):

        x = pyg.x[:, [0, 1, 2]]
        edge_index = pyg.edge_index
        batch = pyg.batch
        num_instances = pyg.num_node_per_example.shape[0]
        edge_index_pc = pyg.edge_index_conjunctions
        edge_index_mc = pyg.edge_index_disjunctions

        node_embed_gin, graph_embed_gin = self.embedding_gin(
            x, edge_index, batch
        )
        node_embed_dghan, graph_embed_dghan = self.embedding_dghan(
            x, add_self_loops(edge_index_pc)[0], add_self_loops(edge_index_mc)[0], num_instances
        )
        node_embed = torch.cat([node_embed_gin, node_embed_dghan], dim=-1)
        graph_embed = torch.cat([graph_embed_gin, graph_embed_dghan], dim=-1)

        return node_embed, graph_embed


class TBGAT(torch.nn.Module):
    def __init__(self,
                 in_channels_fwd,
                 in_channels_bwd,
                 hidden_channels,
                 out_channels,
                 heads=4,
                 drop_out_for_gat=0):
        super().__init__()

        # forward flow embedding
        self.conv_f1 = GATConv(
            in_channels_fwd,
            hidden_channels,
            heads=heads,
            dropout=drop_out_for_gat,
            flow='source_to_target')

        self.conv_f2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            dropout=drop_out_for_gat,
            flow='source_to_target')

        self.conv_f3 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=drop_out_for_gat,
            flow='source_to_target')

        # backward flow embedding
        self.conv_b1 = GATConv(
            in_channels_bwd,
            hidden_channels,
            heads=heads,
            dropout=drop_out_for_gat,
            flow='target_to_source')

        self.conv_b2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            dropout=drop_out_for_gat,
            flow='target_to_source')

        self.conv_b3 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=drop_out_for_gat,
            flow='target_to_source')

    def forward(self, x, edge_index, batch, drop_out=0):
        # add self-loop
        edge_index, _ = add_self_loops(edge_index)

        # x forward
        h_fwd0 = x[:, [0, 1, 3]]  # dur, the earliest starting time, fwd_topo_order

        # forward embedding
        h_fwd1 = F.elu(self.conv_f1(h_fwd0, edge_index))
        h_fwd1 = F.dropout(h_fwd1, p=drop_out, training=self.training)
        h_fwd2 = F.elu(self.conv_f2(h_fwd1, edge_index))
        h_fwd2 = F.dropout(h_fwd2, p=drop_out, training=self.training)
        h_fwd3 = self.conv_f3(h_fwd2, edge_index)

        # x backward
        h_bwd0 = x[:, [0, 2, 4]]  # dur, the latest starting time, bwd_topo_order

        # backward embedding
        h_bwd1 = F.elu(self.conv_b1(h_bwd0, edge_index))
        h_bwd1 = F.dropout(h_bwd1, p=drop_out, training=self.training)
        h_bwd2 = F.elu(self.conv_b2(h_bwd1, edge_index))
        h_bwd2 = F.dropout(h_bwd2, p=drop_out, training=self.training)
        h_bwd3 = self.conv_b3(h_bwd2, edge_index)

        # node embedding
        h_node = torch.cat([h_fwd3, h_bwd3], dim=-1)

        # graph pooling via average pooling
        g_pool = global_mean_pool(h_node, batch)

        return h_node, g_pool


class Actor(torch.nn.Module):

    def __init__(self,
                 # embedding parameters
                 in_channels_fwd,
                 in_channels_bwd,
                 hidden_channels,
                 out_channels,
                 heads=4,
                 dropout_for_gat=0):

        super().__init__()

        if args.embed_net == 'TBGAT':
            self.embedding_network = TBGAT(
                in_channels_fwd=in_channels_fwd,
                in_channels_bwd=in_channels_bwd,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                heads=heads,
                drop_out_for_gat=dropout_for_gat
            )
        else:
            self.embedding_network = TPMCAM()

        # policy
        self.policy = Sequential(
            Linear(out_channels * 8 + 1, out_channels * 2)
            if args.embed_tabu_label else
            Linear(out_channels * 8, out_channels * 2),
            # torch.nn.BatchNorm1d(out_channels * 2),
            torch.nn.Tanh(),
            Linear(out_channels * 2, out_channels * 2),
            # torch.nn.BatchNorm1d(out_channels * 2),
            torch.nn.Tanh(),
            Linear(out_channels * 2, out_channels * 2),
            # torch.nn.BatchNorm1d(out_channels * 2),
            torch.nn.Tanh(),
            Linear(out_channels * 2, 1),
        )

    def forward(self, pyg_sol, feasible_action, optimal_mark, cmax=None, drop_out=0):

        action_list_merged = [actions[0] for actions in feasible_action if actions]

        # if action_list_merged == [] then all instances are optimally solved; return dummy log_p and ent
        if not action_list_merged:

            return None, \
                   torch.zeros(
                       size=[len(feasible_action), 1],
                       dtype=torch.float,
                       device=pyg_sol.edge_index.device,
                       requires_grad=True
                   ), \
                   torch.zeros(
                       size=[len(feasible_action), 1],
                       dtype=torch.float,
                       device=pyg_sol.edge_index.device,
                       requires_grad=True
                   )

        else:
            if args.embed_net == 'TBGAT':
                x = pyg_sol.x
                edge_index = pyg_sol.edge_index
                batch = pyg_sol.batch
                # embedding disjunctive graph...
                node_h, g_pool = self.embedding_network(x, edge_index, batch, drop_out=drop_out)
            elif args.embed_net == 'TPMCAM':
                node_h, g_pool = self.embedding_network(pyg_sol)
            else:
                raise RuntimeError("Not known embed net '{}'.".format(args.embed_net))

            node_h = torch.cat(
                [node_h, g_pool.repeat_interleave(repeats=pyg_sol.num_node_per_example, dim=0)],
                dim=-1
            )

            if args.action_selection_type == 'ls':
                sampled_action, log_prob_padded, entropy_padded = self.move_selector_ls(
                    action_set=feasible_action,
                    node_h=node_h,
                    optimal_mark=optimal_mark
                )
                return sampled_action, log_prob_padded, entropy_padded
            elif args.action_selection_type == 'ts_inner':
                pass
            elif args.action_selection_type == 'ts_outer':
                sampled_action, log_prob_padded, entropy_padded = self.move_selector_ts_outer(
                    node_h=node_h,
                    action_set=feasible_action,
                    optimal_mark=optimal_mark
                )
                return sampled_action, log_prob_padded, entropy_padded
            else:
                raise RuntimeError("Unsupported framework type.")

    def move_selector_ls(self,
                         action_set,
                         node_h,
                         optimal_mark):

        ## compute action probability
        # get action embedding ready
        action_merged_with_tabu_label = torch.cat([actions[0] for actions in action_set if actions], dim=0)
        actions_merged = action_merged_with_tabu_label[:, :2]
        tabu_label = action_merged_with_tabu_label[:, [2]]
        action_h = torch.cat(
            [
                node_h[actions_merged[:, 0]],
                node_h[actions_merged[:, 1]],
                tabu_label
            ],
            dim=-1
        )

        if not args.embed_tabu_label:
            action_h = action_h[:, :-1]

        # compute action score
        action_count = [actions[0].shape[0] for actions in action_set if actions]  # if no action then ignore
        action_score = self.policy(action_h)
        _max_count = max(action_count)
        actions_score_split = list(torch.split(action_score, split_size_or_sections=action_count))
        padded_score = pad_sequence(actions_score_split, padding_value=-torch.inf).transpose(0, -1).transpose(0, 1)

        # sample actions
        pi = F.softmax(padded_score, dim=-1)
        dist = Categorical(probs=pi)
        action_id = dist.sample()
        padded_action = pad_sequence(
            [actions[0][:, :2] for actions in action_set if actions],
        ).transpose(0, 1)
        sampled_action = torch.gather(
            padded_action, index=action_id.repeat(1, 2).view(-1, 1, 2), dim=1
        ).squeeze(dim=1)
        # print(feasible_action)
        # print(action_id)
        # print(sampled_action)

        # greedy action
        # action_id = torch.argmax(pi, dim=-1)

        # compute log_p and policy entropy regardless of optimal sol
        log_prob = dist.log_prob(action_id)
        entropy = dist.entropy()

        # compute padded log_p, where optimal sol has 0 log_0, since no action, otherwise cause shape bug
        log_prob_padded = torch.zeros(
            size=optimal_mark.shape,
            device=action_h.device,
            dtype=torch.float
        )
        log_prob_padded[~optimal_mark, :] = log_prob.squeeze()

        # compute padded ent, where optimal sol has 0 ent, since no action, otherwise cause shape bug
        entropy_padded = torch.zeros(
            size=optimal_mark.shape,
            device=action_h.device,
            dtype=torch.float
        )
        entropy_padded[~optimal_mark, :] = entropy.squeeze()

        return sampled_action, log_prob_padded, entropy_padded

    def move_selector_ts_outer(self,
                               node_h,
                               action_set,
                               optimal_mark):

        # rm tabu move
        action_set_wo_tabu = []
        for actions in action_set:
            if actions:
                cond = actions[0][:, -1] == 1
                if cond.sum() == actions[0].shape[0]:  # if all tabu, then do not rm any move
                    action_set_wo_tabu.append(actions)
                else:
                    action_set_wo_tabu.append([actions[0][~cond, :]])

        # get action embedding ready
        action_merged_with_tabu_label = torch.cat([actions[0] for actions in action_set_wo_tabu], dim=0)
        actions_merged = action_merged_with_tabu_label[:, :2]
        tabu_label = action_merged_with_tabu_label[:, [2]]
        action_h = torch.cat(
            [
                node_h[actions_merged[:, 0]],
                node_h[actions_merged[:, 1]],
                tabu_label
            ],
            dim=-1
        )

        if not args.embed_tabu_label:
            action_h = action_h[:, :-1]

        action_count = [actions[0].shape[0] for actions in action_set_wo_tabu]  # if no action then ignore
        action_score = self.policy(action_h)
        _max_count = max(action_count)
        actions_score_split = list(torch.split(action_score, split_size_or_sections=action_count))
        padded_score = pad_sequence(actions_score_split, padding_value=-torch.inf).transpose(0, -1).transpose(0, 1)

        # sample actions
        pi = F.softmax(padded_score, dim=-1)
        dist = Categorical(probs=pi)
        action_id = dist.sample()
        padded_action = pad_sequence(
            [actions[0][:, :2] for actions in action_set_wo_tabu],
        ).transpose(0, 1)
        sampled_action = torch.gather(
            padded_action, index=action_id.repeat(1, 2).view(-1, 1, 2), dim=1
        ).squeeze(dim=1)
        # print(feasible_action)
        # print(action_id)
        # print(sampled_action)

        # greedy action
        # action_id = torch.argmax(pi, dim=-1)

        # compute log_p and policy entropy regardless of optimal sol
        log_prob = dist.log_prob(action_id)
        entropy = dist.entropy()

        # compute padded log_p, where optimal sol has 0 log_0, since no action, otherwise cause shape bug
        log_prob_padded = torch.zeros(
            size=optimal_mark.shape,
            device=action_h.device,
            dtype=torch.float
        )
        log_prob_padded[~optimal_mark, :] = log_prob.squeeze()

        # compute padded ent, where optimal sol has 0 ent, since no action, otherwise cause shape bug
        entropy_padded = torch.zeros(
            size=optimal_mark.shape,
            device=action_h.device,
            dtype=torch.float
        )
        entropy_padded[~optimal_mark, :] = entropy.squeeze()

        return sampled_action, log_prob_padded, entropy_padded

    def move_selector_ts_inner(self,
                               sol,
                               cmax,
                               action_set,
                               node_h,
                               optimal_mark):

        # sort edge_index otherwise to_data_list() fn will be messed and bug
        sol.edge_index = sort_edge_index(sol.edge_index)
        # sort edge_index_disjunctions otherwise to_data_list() fn will be messed and bug
        sol.edge_index_disjunctions = sort_edge_index(sol.edge_index_disjunctions)

        # copy G for one-step forward
        G_list = sol.to_data_list()
        num_nodes_per_example = torch.tensor([G.num_nodes for G in G_list], device=self.device)
        G_expanded = []
        repeats = []
        action_exist = []
        for _, (a, g) in enumerate(zip(action_set, G_list)):
            if not a:
                G_expanded += [g.clone()]
                repeats += [1]
                action_exist += [False]
            else:
                G_expanded += [g.clone() for _ in range(a[0].shape[0])]
                repeats += [a[0].shape[0]]
                action_exist += [True for _ in range(a[0].shape[0])]
        G_expanded = Batch.from_data_list(G_expanded)
        num_nodes_per_example_one_step = torch.repeat_interleave(
            num_nodes_per_example,
            repeats=torch.tensor(repeats, device=self.device)
        )

        ## prepare actions for one-step rollout
        # for rm operation id increment for old action
        _operation_index_helper1 = torch.cumsum(
            num_nodes_per_example, dim=0
        ) - num_nodes_per_example

        # for add operation id increment for new action
        _operation_index_helper2 = torch.cumsum(
            num_nodes_per_example_one_step, dim=0
        ) - num_nodes_per_example_one_step
        _operation_index_helper2 = _operation_index_helper2[action_exist]

        # merge all action by rm and add action id
        action_merged_one_step = torch.cat(
            [actions[0][:, :2] - _operation_index_helper1[_] for _, actions in enumerate(action_set) if
             actions], dim=0
        ) + _operation_index_helper2.unsqueeze(-1)

        ## one step
        # action: u -> v
        u = action_merged_one_step[:, 0]
        v = action_merged_one_step[:, 1]
        edge_index_disjunctions = G_expanded.edge_index_disjunctions
        # mask for arcs: m^{-1}(u) -> u, u -> v, and v -> m(v), all have shape [num edge]
        mask1 = (~torch.eq(edge_index_disjunctions[1], u.reshape(-1, 1))).sum(dim=0) == u.shape[0]  # m^{-1}(u) -> u
        mask2 = (~torch.eq(edge_index_disjunctions[0], u.reshape(-1, 1))).sum(dim=0) == u.shape[0]  # u -> v
        mask3 = (~torch.eq(edge_index_disjunctions[0], v.reshape(-1, 1))).sum(dim=0) == v.shape[0]  # v -> m(v)
        # edges to be removed
        edge_m_neg_u_to_u = edge_index_disjunctions[:, ~mask1]
        edge_u_to_v = edge_index_disjunctions[:, ~mask2]
        edge_v_to_mv = edge_index_disjunctions[:, ~mask3]
        # remove arcs: m^{-1}(u) -> u, u -> v, and v -> m(v)
        mask = mask1 * mask2 * mask3
        edge_index_disjunctions = edge_index_disjunctions[:, mask]
        # build new arcs m^{-1}(u) -> v
        _idx_m_neg_u_to_v = torch.eq(edge_m_neg_u_to_u[1].unsqueeze(1), edge_u_to_v[0]).nonzero()[:, 1]
        _edge_m_neg_u_to_v = torch.stack([edge_m_neg_u_to_u[0], edge_u_to_v[1, _idx_m_neg_u_to_v]])
        # build new arcs v -> u
        _edge_v_to_u = torch.flip(edge_u_to_v, dims=[0])
        # build new arcs u -> m(v)
        _idx_u_to_mv = torch.eq(edge_v_to_mv[0].unsqueeze(1), edge_u_to_v[1]).nonzero()[:, 1]
        _edge_u_to_mv = torch.stack([edge_u_to_v[0, _idx_u_to_mv], edge_v_to_mv[1]])
        # add new arcs to edge_index_disjunctions
        edge_index_disjunctions = torch.cat(
            [edge_index_disjunctions, _edge_m_neg_u_to_v, _edge_v_to_u, _edge_u_to_mv], dim=1
        )  # unsorted

        # Cmax before one step
        Cmax_before_one_step = cmax.repeat_interleave(
            repeats=torch.tensor(repeats, device=self.device)
        )

        # Cmax after one step
        G_expanded.edge_index = torch.cat([edge_index_disjunctions, G_expanded.edge_index_conjunctions], dim=1)
        _, _, Cmax_after_one_step, _, _, _ = self.evaluator.eval(
            G_expanded,
            num_nodes_per_example=num_nodes_per_example_one_step
        )

        # compute tabu label
        tabu_label_split = [actions[0][:, 2].bool() for actions in action_set if actions]

        # select action
        splits_counts = [tb.shape[0] for tb in tabu_label_split]
        action_set_wo_empty = [[action[0][:, :2]] for action in action_set if action]
        Cmax_before_split = list(
            torch.split(Cmax_before_one_step[action_exist], split_size_or_sections=splits_counts)
        )
        Cmax_after_split = list(
            torch.split(Cmax_after_one_step[action_exist], split_size_or_sections=splits_counts)
        )
        selected_actions = []
        for idx, (tb_label, Cmax_before, Cmax_after, action) in enumerate(
                zip(tabu_label_split, Cmax_before_split, Cmax_after_split, action_set_wo_empty)
        ):
            action = action[0]  # get action tensor from list

            aspiration_flag = torch.lt(Cmax_after, Cmax_before).long()
            if (~tb_label).sum() == 0 and aspiration_flag.sum() == 0:  # random select
                selected_a = random.choice([*action])
                selected_actions.append(selected_a)
            elif (~tb_label).sum() != 0:
                if self.if_drl:
                    sampled_action, _, _ = self.drl_agent(
                        pyg_sol=sol,
                        feasible_action=[[torch.cat([action, tb_label.unsqueeze(1)], dim=1)[~tb_label, :]]],
                        optimal_mark=optimal_mark
                    )
                    selected_a = sampled_action[0]
                else:
                    Cmax_after_non_tabu = Cmax_after[~tb_label]
                    action_index = Cmax_after_non_tabu.argmin(dim=0)
                    selected_a = action[~tb_label, :][action_index]
                selected_actions.append(selected_a)
            else:
                if self.if_drl:
                    sampled_action, _, _ = self.drl_agent(
                        pyg_sol=sol,
                        feasible_action=[[torch.cat([action, tb_label.unsqueeze(1)], dim=1)[
                                          torch.where(aspiration_flag == 1)[0], :]]],
                        optimal_mark=optimal_mark
                    )
                    selected_a = sampled_action[0]
                else:
                    action_index = random.choice([*torch.where(aspiration_flag == 1)[0]])
                    selected_a = action[action_index]
                selected_actions.append(selected_a)

        selected_actions = torch.stack(selected_actions)

        return selected_actions


if __name__ == '__main__':
    import time
    import random
    from env.generateJSP import uni_instance_gen
    from env.environment import Env

    # j, m, batch_size = {'low': 100, 'high': 101}, {'low': 20, 'high': 21}, 500
    # j, m, batch_size = {'low': 30, 'high': 31}, {'low': 20, 'high': 21}, 64
    # j, m, batch_size = {'low': 10, 'high': 11}, {'low': 10, 'high': 11}, 128
    # j, m, batch_size = {'low': 20, 'high': 21}, {'low': 15, 'high': 16}, 64
    j, m, batch_size = {'low': 10, 'high': 11}, {'low': 10, 'high': 11}, 3
    # [j, m], batch_size = [np.array(
    #     [[15, 15],  # Taillard
    #      [20, 15],
    #      [20, 20],
    #      [30, 15],
    #      [30, 20],
    #      [50, 15],
    #      [50, 20],
    #      [100, 20],
    #      [10, 10],  # ABZ, not include: 20x15
    #      [6, 6],  # FT, not include: 10x10
    #      [20, 5],
    #      [10, 5],  # LA, not include: 10x10, 15x15
    #      [15, 5],
    #      [20, 5],
    #      [15, 10],
    #      [20, 10],
    #      [30, 10],
    #      [50, 10],  # SWV, not include: 20x10, 20x15
    #      # no ORB 10x10, no YN 20x20
    #      ])[:, _] for _ in range(2)], 128

    l = 1
    h = 99
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_type = 'fdd-divide-wkr'  # 'spt', 'fdd-divide-mwkr'
    seed = 25  # 6: two paths for the second instance
    np.random.seed(seed)
    backward_option = False  # if true usually do not pass N5 property
    print_step_time = True
    print_time_of_calculating_moves = True
    print_action_space_compute_time = True
    path_finder = 'pytorch'  # 'networkx' or 'pytorch'
    tb_size = -1  # tabu_size
    torch.random.manual_seed(seed)
    random.seed(seed)

    if type(j) is dict and type(m) is dict:  # random range
        insts = [np.concatenate(
            [uni_instance_gen(n_j=np.random.randint(**j), n_m=np.random.randint(**m), low=l, high=h)]
        ) for _ in range(batch_size)]
    else:  # random from set
        insts = []
        for _ in range(batch_size):
            i = random.randint(0, j.shape[0] - 1)
            inst = np.concatenate(
                [uni_instance_gen(n_j=j[i], n_m=m[i], low=l, high=h)]
            )
            insts.append(inst)

    # insts = np.load('../test_data_jssp/tai20x15.npy')
    # print(insts)

    env = Env()
    G, (feasible_a, mark, paths) = env.reset(
        instances=insts,
        init_sol_type=init_type,
        tabu_size=tb_size,
        device=dev,
        mask_previous_action=backward_option,
        longest_path_finder=path_finder
    )

    env.cpm_eval()

    # print(env.instance_size)

    net = Actor(
        in_channels_fwd=3,
        in_channels_bwd=3,
        hidden_channels=128,
        out_channels=128,
        heads=4,
        dropout_for_gat=0
    ).to(dev)

    data = []
    h_embd = None
    g_embd = None
    log_p = None
    ent = None
    for _ in range(5):
        t1_ = time.time()
        print('step {}'.format(_))

        sampled_a, log_p, ent = net(
            pyg_sol=G,
            feasible_action=feasible_a,
            optimal_mark=mark,
            cmax=env.current_objs
        )

        G, reward, (feasible_a, mark, paths) = env.step(
            action=sampled_a,
            prt=print_step_time,
            show_action_space_compute_time=print_action_space_compute_time
        )
        # print(env.current_objs)
        # print(env.incumbent_objs)
        t2_ = time.time()
        print("This iteration takes: {:.4f}".format(t2_ - t1_))
        print()

    env.cpm_eval()

    loss = log_p.mean()
    grad = torch.autograd.grad(loss + torch.tensor(1., requires_grad=True), [param for param in net.parameters()])

    log_p_normal = log_p.clone()
    # print(log_p_normal)

    # parameter after backward with normal log_p
    # import torch.optim as optim
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # print(log_p_normal)
    # loss = log_p_normal.mean()
    # # backward
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print([param for param in net.parameters()])

    # parameter after backward with mean of dummy log_p and normal log_p, should be equal with that of normal log_p,
    # since dummy log_p affect nothing
    # sampled_a, log_p_dummy, ent_dummy = net(
    #     pyg_sol=G,
    #     feasible_action=[[], [], []],
    #     optimal_mark=mark,
    #     critical_path=paths
    # )
    # import torch.optim as optim
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # loss = torch.cat([log_p_dummy, log_p_normal], dim=-1).sum(dim=-1)
    # # backward
    # optimizer.zero_grad()
    # loss.mean().backward()
    # optimizer.step()
    # print([param for param in net.parameters()])
