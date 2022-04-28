import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch.nn.utils.rnn import pad_sequence


class Embed(torch.nn.Module):
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
                 dropout_for_gat=0,
                 # policy parameters
                 policy_l=3):
        super().__init__()

        self.embedding_network = Embed(
            in_channels_fwd=in_channels_fwd,
            in_channels_bwd=in_channels_bwd,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            drop_out_for_gat=dropout_for_gat
        )

        # policy
        self.policy = Sequential(
            Linear(out_channels * 8 + 1, out_channels * 2),
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

    def forward(self, pyg_sol, feasible_action, optimal_mark, critical_path=None, drop_out=0):
        x = pyg_sol.x
        edge_index = pyg_sol.edge_index
        batch = pyg_sol.batch

        # embedding disjunctive graph...
        node_h, g_pool = self.embedding_network(x, edge_index, batch, drop_out=drop_out)
        node_h = torch.cat(
            [node_h, g_pool.repeat_interleave(repeats=pyg_sol.num_node_per_example, dim=0)],
            dim=-1
        )

        ## compute action probability
        # get action embedding ready
        action_merged_with_tabu_label = torch.cat([actions[0] for actions in feasible_action if actions], dim=0)
        actions_merged = action_merged_with_tabu_label[:, :2]
        tabu_label = action_merged_with_tabu_label[:, [2]]
        action_h_with_tabu_label = torch.cat(
            [node_h[actions_merged[:, 0]], node_h[actions_merged[:, 1]], tabu_label], dim=-1
        )
        # compute action score
        action_score = self.policy(action_h_with_tabu_label)
        action_count = [actions[0].shape[0] for actions in feasible_action if actions]
        _max_count = max(action_count)
        actions_score_split = list(torch.split(action_score, split_size_or_sections=action_count))
        padded_score = pad_sequence(actions_score_split, padding_value=-torch.inf).transpose(0, -1).transpose(0, 1)
        # sample actions
        pi = F.softmax(padded_score, dim=-1)
        dist = Categorical(probs=pi)
        action_id = dist.sample()
        padded_action = pad_sequence(
            [actions[0][:, :2] for actions in feasible_action if actions],
        ).transpose(0, 1)
        sampled_action = torch.gather(
            padded_action, index=action_id.repeat(1, 2).view(-1, 1, 2), dim=1
        ).squeeze(dim=1)
        # action_id = torch.argmax(pi, dim=-1)  # greedy action

        # compute log_p and policy entropy regardless of optimal sol
        log_prob = dist.log_prob(action_id)
        entropy = dist.entropy()

        # compute padded log_p, where optimal sol has 0 log_0, since no action, otherwise cause shape bug
        log_prob_padded = torch.zeros(
            size=optimal_mark.shape,
            device=x.device,
            dtype=torch.float
        )
        log_prob_padded[~optimal_mark, :] = log_prob.squeeze()

        # compute padded ent, where optimal sol has 0 ent, since no action, otherwise cause shape bug
        entropy_padded = torch.zeros(
            size=optimal_mark.shape,
            device=x.device,
            dtype=torch.float
        )
        entropy_padded[~optimal_mark, :] = entropy.squeeze()

        return sampled_action, log_prob_padded, entropy_padded


if __name__ == '__main__':
    import time
    import random
    from env.generateJSP import uni_instance_gen
    from env.environment import Env

    # j, m, batch_size = {'low': 100, 'high': 101}, {'low': 20, 'high': 21}, 500
    # j, m, batch_size = {'low': 30, 'high': 31}, {'low': 20, 'high': 21}, 64
    # j, m, batch_size = {'low': 10, 'high': 11}, {'low': 10, 'high': 11}, 128
    # j, m, batch_size = {'low': 20, 'high': 21}, {'low': 15, 'high': 16}, 64
    j, m, batch_size = {'low': 3, 'high': 6}, {'low': 3, 'high': 6}, 3
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
    init_type = 'fdd-divide-mwkr'  # 'spt', 'fdd-divide-mwkr'
    seed = 25  # 6: two paths for the second instance
    np.random.seed(seed)
    backward_option = False  # if true usually do not pass N5 property
    print_step_time = True
    print_time_of_calculating_moves = True
    print_action_space_compute_time = True
    path_finder = 'pytorch'  # 'networkx' or 'pytorch'
    tb_size = 20  # tabu_size
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

    # insts = np.load('../test_data/tai20x15.npy')
    # print(insts)

    env = Env()
    G, (action_set, mark, paths) = env.reset(
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
            feasible_action=action_set,
            optimal_mark=mark,
            critical_path=paths
        )

        G, reward, (action_set, mark, paths) = env.step(
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
    grad = torch.autograd.grad(loss, [param for param in net.parameters()])
