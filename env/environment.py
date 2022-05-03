import os
import sys
import time

import torch_geometric.utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from env.left_shift import permissibleLeftShift
import torch
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch
from env.message_passing_evl import MassagePassingEval
import networkx as nx
import random
from env.message_passing_evl import cpm_forward_and_backward


def index_to_mask(index, size=None):
    """
    index: node index
    size: total number of nodes
    return: [total number of nodes，] where node index = True
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


class Env:
    def __init__(
            self,
            dur_norm=99,
            est_lst_norm=1000,
            mask_previous_action=False,  # whether consider previous selected action
            longest_path_finder='pytorch'  # 'networkx' or 'pytorch'
    ):

        self.itr = 0
        self.dur_norm = dur_norm
        self.est_lst_norm = est_lst_norm
        self.evaluator = MassagePassingEval()
        self.instances = None
        self.instance_size = None
        self.incumbent_objs = None
        self.current_objs = None
        self.G_batch = None  # pyg disjunctive graph
        self.job_count = None
        self.machine_count = None
        self.size = None
        self._machine_count_cumsum = None
        self.num_nodes_per_example = None
        self.S = None  # all S
        self.T = None  # all T
        self.num_instance = None
        self.tabu_list = None
        self.tabu_size = None
        self.previous_action = None  # do not select previous action
        # do not consider backward move if True, e.g., [5, 8] will be excluded if previous move is [8, 5], default True.
        self.mask_previous_action = mask_previous_action
        self.longest_path_finder = longest_path_finder

    def _init_solver(self, init, device, p_lists=None, prt=False):
        init, device = init, device

        Gs = []
        for _, instance in enumerate(self.instances):

            ## preparation
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs, n_machines = dur_mat.shape[0], dur_mat.shape[1]
            n_operations = n_jobs * n_machines
            last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
            # initialize action space: [n_jobs,], the first column
            candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, 0]
            # initialize the mask: [n_jobs,]
            mask = np.zeros(shape=n_jobs, dtype=bool)
            # initialize gantt chart: [n_machines, n_jobs]; The multiplier can be anything, here we use -1
            gant_chart = - np.ones_like(dur_mat.transpose(), dtype=np.int32)
            # initialize operation id on machines: [n_machines, n_jobs]
            opIDsOnMchs = - n_jobs * np.ones_like(dur_mat.transpose(), dtype=np.int32)
            # initialize operation finished mask: [n_jobs, n_machines]
            finished_mark = np.zeros_like(mch_mat, dtype=np.int32)
            # initialize action
            actions = []

            ## construct adj for precedent constraint
            # Create adjacent matrix for precedence constraints
            adj_mat_pc = np.eye(n_operations, k=-1, dtype=int)
            # first column does not have upper stream conj_nei
            adj_mat_pc[np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, 0]] = 0
            # pad dummy S and T nodes
            adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)
            # connect S with 1st operation of each job
            adj_mat_pc[[_ for _ in range(1, n_operations + 2 - 1, n_machines)], 0] = 1
            # connect last operation of each job to T
            adj_mat_pc[-1, [_ for _ in range(n_machines, n_operations + 2 - 1, n_machines)]] = 1
            # convert input adj from column pointing to row, to, row pointing to column
            adj_mat_pc = np.transpose(adj_mat_pc)

            ## construct adjacent matrix for machine clique
            if init in ['spt', 'fdd-divide-wkr']:  # if using dispatching rule
                adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)
                for _ in range(n_operations):

                    if init == 'spt':
                        candidate_masked = candidate_oprs[np.where(~mask)]
                        dur_candidate = np.take(dur_mat, candidate_masked)
                        idx = np.random.choice(np.where(dur_candidate == np.min(dur_candidate))[0])
                        action = candidate_masked[idx]
                    elif init == 'fdd-divide-wkr':
                        candidate_masked = candidate_oprs[np.where(~mask)]
                        fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
                        wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
                        priority = fdd / wkr
                        idx = np.random.choice(np.where(priority == np.min(priority))[0])
                        action = candidate_masked[idx]
                    else:
                        raise RuntimeError('Rule must be one of "spt" or "fdd-divide-wkr".')
                    actions.append(action)

                    permissibleLeftShift(
                        a=action,
                        durMat=dur_mat,
                        mchMat=mch_mat,
                        mchsStartTimes=gant_chart,
                        opIDsOnMchs=opIDsOnMchs
                    )

                    # update action space or mask
                    if action not in last_col:
                        candidate_oprs[action // n_machines] += 1
                    else:
                        mask[action // n_machines] = 1
                    # update finished_mark:
                    finished_mark[action // n_machines, action % n_machines] = 1

                for i in range(opIDsOnMchs.shape[1] - 1):
                    adj_mat_mc[opIDsOnMchs[:, i + 1], opIDsOnMchs[:, i]] = 1
                # add S and T to machine clique adj
                adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)
                # convert input adj from column pointing to row, to, row pointing to column
                adj_mat_mc = np.transpose(adj_mat_mc)
            elif init == 'plist':
                # construct NIPS adjacent matrix only for machine cliques
                if p_lists is None:
                    plist = np.random.permutation(np.arange(instance.shape[1]).repeat(instance.shape[2]))
                else:
                    plist = p_lists[_]
                ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()  # Init operations mat
                list_for_latest_task_on_machine = [None] * n_machines  # Init list_for_latest_task_on_machine
                # create adjacent matrix for machine clique
                adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)
                for job_id in plist:
                    op_id = ops_mat[job_id][0]
                    m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                    if list_for_latest_task_on_machine[m_id_for_action] is not None:
                        adj_mat_mc[op_id, list_for_latest_task_on_machine[m_id_for_action]] = 1
                    list_for_latest_task_on_machine[m_id_for_action] = op_id
                    ops_mat[job_id].pop(0)
                # add S and T to machine clique adj
                adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)
                # convert input adj from column pointing to row, to, row pointing to column
                adj_mat_mc = np.transpose(adj_mat_mc)
            else:
                raise RuntimeError('Not support initial solver type {}.'.format(init))

            ## adj and node attribute for disjunctive graph
            adj_all = adj_mat_pc + adj_mat_mc
            # S, T has duration 0
            duration = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0)
            # S, T has machine -1
            machine_id = np.pad(mch_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=-1)

            ## save to pyg data
            Gs.append(
                Data(dur=torch.from_numpy(duration),
                     m_id=torch.from_numpy(machine_id),
                     num_nodes=duration.shape[0],
                     edge_index=torch.nonzero(torch.from_numpy(adj_all)).t().contiguous(),
                     edge_index_conjunctions=torch.nonzero(torch.from_numpy(adj_mat_pc)).t().contiguous(),
                     edge_index_disjunctions=torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous())
            )

        G_batch = Batch.from_data_list(Gs).to(device)

        ## calculating est, lst using message-passing
        t_ = time.time()
        est, lst, make_span, count, fwd_topo_batches, bwd_topo_batches = self.evaluator.eval(
            G_batch, num_nodes_per_example=self.num_nodes_per_example)
        if prt:
            print('Message-passing takes: {:.5f}'.format(time.time() - t_))

        ## update graph node features to include normalized est, lst, and dur
        dur_fea = G_batch.dur / self.dur_norm
        est_fea = est.unsqueeze(1) / self.est_lst_norm
        lst_fea = lst.unsqueeze(1) / self.est_lst_norm
        fwd_topo_fea = self.fwd_topological_feature(fwd_topo_batches, G_batch.ptr)
        bwd_topo_fea = self.bwd_topological_feature(bwd_topo_batches, G_batch.ptr)
        x = torch.cat([dur_fea, est_fea, lst_fea, fwd_topo_fea, bwd_topo_fea], dim=1)
        G_batch.x = x
        G_batch.__setattr__("est", est)
        G_batch.__setattr__("num_node_per_example", self.num_nodes_per_example)

        return G_batch, make_span, count

    @staticmethod
    def fwd_topological_feature(fwd_topo_batches, ptr):
        fwd_topo_order = torch.cat(fwd_topo_batches, dim=0).float()
        node_index_boundary_left = ptr[:-1]
        node_index_boundary_right = ptr[1:]
        topo_order_all = []
        topo_order_feature = []
        for node_boundary_left, node_boundary_right in zip(node_index_boundary_left, node_index_boundary_right):
            topo_sort = fwd_topo_order[(node_boundary_left <= fwd_topo_order) * (fwd_topo_order < node_boundary_right)]
            topo_sort_each = topo_sort - node_boundary_left
            topo_order_all.append(topo_sort_each)
            topo_feature = torch.argsort(topo_sort_each)
            topo_order_feature.append(topo_feature / topo_feature.max())
        topo_order_feature = torch.cat(topo_order_feature, dim=0).unsqueeze(-1)
        return topo_order_feature

    @staticmethod
    def bwd_topological_feature(bwd_topo_batches, ptr):
        bwd_topo_order = torch.cat(bwd_topo_batches, dim=0).float()
        node_index_boundary_left = ptr[:-1]
        node_index_boundary_right = ptr[1:]
        topo_order_all = []
        topo_order_feature = []
        for node_boundary_left, node_boundary_right in zip(node_index_boundary_left, node_index_boundary_right):
            topo_sort = bwd_topo_order[(node_boundary_left <= bwd_topo_order) * (bwd_topo_order < node_boundary_right)]
            topo_sort_each = topo_sort - node_boundary_left
            topo_order_all.append(topo_sort_each)
            topo_feature = torch.argsort(topo_sort_each)
            topo_order_feature.append(topo_feature / topo_feature.max())
        topo_order_feature = torch.cat(topo_order_feature, dim=0).unsqueeze(-1)
        return topo_order_feature

    def get_candidate_moves(self, ns_type='N5', prt=False):
        t0 = time.time()

        if self.longest_path_finder == 'networkx':
            node_all = torch.where(
                torch.eq(self.G_batch.x[:, 1], self.G_batch.x[:, 2])  # est == lst
            )[0]
            # sub graph containing all critical nodes
            sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                subset=node_all,
                edge_index=self.G_batch.edge_index,
                edge_attr=-self.G_batch.dur[self.G_batch.edge_index[0]]  # shortest path with -edge.w = the longest path
            )
            sub_pyg = Data(edge_index=sub_edge_index, edge_attr=sub_edge_attr, num_nodes=node_all.shape[0])

            # find all longest paths using networkx
            sub_nxg = torch_geometric.utils.to_networkx(sub_pyg, edge_attrs=['edge_attr'], remove_self_loops=True)
            _t1 = time.time()
            paths_all = []
            for i, (s, t) in enumerate(zip(self.S, self.T)):
                longest_paths = list(
                    nx.all_shortest_paths(sub_nxg, source=s, target=t, weight='edge_attr', method='bellman-ford'))
                longest_paths = sorted(longest_paths)
                # print(longest_paths)
                first_path = longest_paths[0]  # always select the first path
                paths_all.append([first_path])
                assert torch.equal(self.G_batch.dur[first_path].sum().float(),
                                   self.current_objs[i])  # check indeed longest

                '''for p in longest_paths:  # consider all longest_paths
                    assert torch.equal(self.G_batch.dur[p].sum().float(),self.current_objs[i])  # check indeed longest
                paths_all.append([p for p in longest_paths])'''
            _t2 = time.time()
        elif self.longest_path_finder == 'pytorch':
            _t1 = time.time()
            paths_all = self.longest_paths()
            paths_all = [sorted([p.cpu().numpy().tolist() for p in ps]) for ps in paths_all]
            # print(paths_all)
            paths_all = [[ps[0]] for ps in paths_all]  # always select the first path
            _t2 = time.time()
        else:
            raise RuntimeError('Not support longest path finder {}.'.format(self.longest_path_finder))

        # calculate critical blocks
        _t3 = time.time()
        critical_operation = torch.tensor(
            sum([p[1:-1] for ps in paths_all for p in ps], []),
            dtype=torch.long, device=self.G_batch.edge_index.device)  # 1-D tensor
        machine_id = self.G_batch.m_id[critical_operation].squeeze()
        instance_id = self.G_batch.batch[critical_operation].squeeze()
        _machine_id_augmented = machine_id + self._machine_count_cumsum[
            critical_operation]  # _machine_id_augmented: 0, 1, 2, ...
        _, _block_size = torch.unique_consecutive(_machine_id_augmented, return_counts=True)
        blk_id = torch.repeat_interleave(
            torch.arange(_block_size.shape[0], device=_block_size.device), _block_size, dim=0)
        blk_joint = torch.stack([critical_operation, blk_id, _machine_id_augmented, instance_id]).t()
        # split blocks according to instance
        _, _count_instance = torch.unique_consecutive(instance_id, return_counts=True)
        _blk_instance = torch.split(blk_joint, split_size_or_sections=_count_instance.tolist(), dim=0)
        # split blocks of each instance into individual block
        blk_split = []
        for blk_instance in _blk_instance:
            _, _count_machine = torch.unique_consecutive(blk_instance[:, 1], return_counts=True)
            blk_split.append(list(torch.split(blk_instance, split_size_or_sections=_count_machine.tolist(), dim=0)))
        _t4 = time.time()

        _t5 = time.time()
        if ns_type == 'N5':  # calculate N5 pairs
            moves = [[] for _ in range(self.num_instance)]
            for instance_id, blks in enumerate(blk_split):
                if len(blks) == 1:  # single blk then no pairs
                    pass
                else:
                    ### first block
                    if blks[0].shape[0] >= 2:
                        moves[instance_id].append(blks[0][-2:, 0])
                    ### middle blocks
                    for b in blks[1:-1]:
                        ## if the block has more than 2 operations
                        if b.shape[0] > 2:
                            moves[instance_id].append(b[:2, 0])
                            moves[instance_id].append(b[-2:, 0])
                        elif b.shape[0] == 2:
                            moves[instance_id].append(b[:2, 0])
                        else:
                            pass
                    ### last block
                    if blks[-1].shape[0] >= 2:
                        moves[instance_id].append(blks[-1][:2, 0])
            merge_move = [[torch.stack(mv, dim=0)] if len(mv) > 0 else [] for mv in moves]
            # print(merge_move)

            # remove previous selected actions if required
            if self.mask_previous_action:
                for instance_id, mv in enumerate(merge_move):
                    if not mv:
                        pass
                    else:
                        mask = torch.isin(mv[0], self.previous_action[instance_id]).sum(1) == 0  # False for remove
                        if mask.sum() != 0:  # if there is something to pick
                            mv[0] = mv[0][mask, :]
                        else:  # if all remove
                            merge_move[instance_id] = []
            # tabu label: 1 tabu, 0 non-tabu
            for instance_id, mv in enumerate(merge_move):
                if mv:
                    tabu_label = (
                        torch.eq(
                            mv[0].unsqueeze(1).repeat(1, self.tabu_list[instance_id].shape[0], 1),
                            self.tabu_list[instance_id]).sum(-1) == 2
                    ).any(dim=1).reshape(-1, 1)
                    mv[0] = torch.cat([mv[0], tabu_label], dim=-1)

            # bool for optimal sol, Ture: the current sol for the corresponding instance is optimal. N5 has property
            # (proposition 1) that when |N5| = 0, then optimal.
            optimal_mark = ~np.array([[len(mv)] for mv in merge_move], dtype=bool)

            _t6 = time.time()

            if prt:
                print('Calculating critical paths takes {:.4f}'.format(_t2 - _t1))
                print('Calculating critical block takes {:.4f}'.format(_t4 - _t3))
                print('Calculating N5 moves with tabu takes {:.4f}'.format(_t6 - _t5))
                print('Total time for calculating N5 moves: {:.4f}'.format(time.time() - t0))
                # print('Total number of paths: {} for total {} instances'.format(
                #     sum([1 for ps in paths_all for _ in ps]),
                #     self.num_instance))
                # print('Block joint:\n', blk_joint)
                # for i, blks in enumerate(blk_split):
                #     print('Critical blocks for instance {}:'.format(i))
                #     for b in blks:
                #         print(b)

        else:
            raise NotImplementedError('Not support other NS yet.')

        return merge_move, optimal_mark, paths_all

    def random_action(self, non_tabu_only=False, show_action_space_compute_time=False):
        candidate_moves, _, _ = self.get_candidate_moves(prt=show_action_space_compute_time)
        selected_moves = []
        for mvs in candidate_moves:
            if len(mvs) != 0:
                mvs = mvs[0]
                if non_tabu_only:
                    sub_mvs = mvs[:, :2][mvs[:, -1].bool()]
                    selected_moves.append(random.choice([*sub_mvs]))
                else:
                    selected_moves.append(random.choice([*mvs[:, :2]]))
        return selected_moves

    def step(self, action=None, ns_type='N5', prt=False, show_action_space_compute_time=False):
        """
        action: [num actions, 2], joint actions for all instances
        """
        if ns_type == 'N5':
            t1_ = time.time()
            # if no action provided, then random generate some
            edge_index_disjunctions = self.G_batch.edge_index_disjunctions
            if action is None:  # if there are actions to select, random select, otherwise optimal and do nothing
                action = self.random_action(show_action_space_compute_time=False)
                if not action:  # if all optimal do nothing
                    self.itr += 1
                    return self.G_batch, \
                           (self.incumbent_objs - self.incumbent_objs).unsqueeze(-1), \
                           self.get_candidate_moves(prt=show_action_space_compute_time)
                action = torch.stack(action)
            t2_ = time.time()
            # print(t2_ - t1_)

            # action: u -> v
            u = action[:, 0]
            v = action[:, 1]
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
            t3_ = time.time()
            # print(t3_ - t2_)

            ## update pygs
            # update edge_index_disjunctions
            self.G_batch.edge_index_disjunctions = edge_index_disjunctions
            self.G_batch.edge_index = torch.cat([edge_index_disjunctions, self.G_batch.edge_index_conjunctions], dim=1)
            # update x
            est, lst, make_span, count, fwd_topo_batches, bwd_topo_batches = self.evaluator.eval(
                self.G_batch,
                num_nodes_per_example=self.num_nodes_per_example)
            dur_fea = self.G_batch.dur / self.dur_norm  # any G will do
            est_fea = est.unsqueeze(1) / self.est_lst_norm
            lst_fea = lst.unsqueeze(1) / self.est_lst_norm
            fwd_topo_fea = self.fwd_topological_feature(fwd_topo_batches, self.G_batch.ptr)
            bwd_topo_fea = self.bwd_topological_feature(bwd_topo_batches, self.G_batch.ptr)
            x = torch.cat([dur_fea, est_fea, lst_fea, fwd_topo_fea, bwd_topo_fea], dim=1)
            self.G_batch.x = x
            self.G_batch.__setattr__("est", est)

            # reward
            reward = torch.where(
                self.incumbent_objs - make_span > 0,
                self.incumbent_objs - make_span,
                torch.tensor(0, dtype=torch.float32, device=make_span.device)
            )
            # update objs
            self.incumbent_objs = torch.minimum(self.incumbent_objs, make_span)
            self.current_objs = make_span
            # update tabu list and do_not_select
            action_instance_id = self.G_batch.batch[u].cpu().numpy()
            for a_reversed, instance_id in zip(action.flip([1]), action_instance_id):
                # update tabu list
                self.tabu_list[instance_id] = torch.cat(
                    [self.tabu_list[instance_id], a_reversed.unsqueeze(0)],
                    dim=0)[1:]
                # do_not_select
                self.previous_action[instance_id] = a_reversed

            self.itr += 1

            t4_ = time.time()
            # print(t4_ - t3_)
            if prt:
                print("Step takes {:.4f} for N5 for {} instances.".format(t4_ - t1_, self.num_instance))

            return self.G_batch, reward.unsqueeze(-1), self.get_candidate_moves(prt=show_action_space_compute_time)
        else:
            raise NotImplementedError('Not support for other NS yet.')

    def cpm_eval(self):
        makespan_cpm = []
        num_nodes_per_example = self.num_nodes_per_example.cpu().numpy()
        end = num_nodes_per_example.cumsum()
        start = end - num_nodes_per_example
        start_end = np.stack([start, end]).transpose()
        for left, right in start_end:
            sub_edge, _ = torch_geometric.utils.subgraph(subset=list(np.arange(left, right)),
                                                         edge_index=self.G_batch.edge_index)
            pyg = Data(edge_index=sub_edge - left, weight=self.G_batch.dur[sub_edge[0]], num_nodes=right - left)
            nxg = torch_geometric.utils.to_networkx(pyg, edge_attrs=['weight'], remove_self_loops=True)
            earliest_start_time, latest_start_time, makespan = cpm_forward_and_backward(nxg)
            makespan_cpm.append(makespan)
        makespan_cpm = torch.tensor(makespan_cpm, dtype=torch.float, device=self.current_objs.device)
        # print(makespan_cpm)
        if torch.equal(makespan_cpm, self.current_objs):
            print("Pass validation with CPM.")
        else:
            print("Not pass validation with CPM.")

    @staticmethod
    def edges_start_from_given_nodes(given_nodes, edge_index, total_num_nodes_in_graph):
        _node_mask = index_to_mask(given_nodes, size=total_num_nodes_in_graph)
        _edge_mask = _node_mask[edge_index[0]]
        connects = edge_index[:, _edge_mask]
        return connects

    @staticmethod
    def edges_end_at_given_nodes(given_nodes, edge_index, total_num_nodes_in_graph):
        _node_mask = index_to_mask(given_nodes, size=total_num_nodes_in_graph)
        _edge_mask = _node_mask[edge_index[1]]
        connects = edge_index[:, _edge_mask]
        return connects

    def longest_paths(self):
        """
        return: all longest paths for each G in self.G_batch
        """
        dur = self.G_batch.dur.squeeze().float()
        est = self.G_batch.est.float()
        edge_index = self.G_batch.edge_index
        num_nodes = self.G_batch.num_nodes
        S = torch.from_numpy(self.S).to(est.device)
        T = torch.from_numpy(self.T).to(est.device)
        u, v = connects = self.edges_start_from_given_nodes(
            given_nodes=S,
            edge_index=edge_index,
            total_num_nodes_in_graph=num_nodes)
        critical_edge_mask = (dur[u] + est[u]) == est[v]
        critical_u, critical_v = critical_edges = connects[:, critical_edge_mask]
        critical_paths = critical_edges
        collect_path = [[] for _ in range(self.num_instance)]
        # print(self.S)
        while critical_edges.shape[1] != 0:
            # print(T)
            u, v = connects = self.edges_start_from_given_nodes(
                given_nodes=critical_v,
                edge_index=edge_index,
                total_num_nodes_in_graph=num_nodes)
            critical_edge_mask = (dur[u] + est[u]) == est[v]
            critical_u, critical_v = critical_edges = connects[:, critical_edge_mask]

            # rearrange paths to prepare for connecting new edges
            rearrange_mask_each_critical_u = torch.eq(critical_u.unsqueeze(1), critical_paths[-1])
            rearrange_idx_for_path = rearrange_mask_each_critical_u.nonzero()[:, 1]
            rearranged_critical_paths = critical_paths[:, rearrange_idx_for_path]
            # rearrange critical edges to prepare for connecting rearranged paths
            critical_edge_repeat = rearrange_mask_each_critical_u.sum(1)
            rearranged_critical_edges = torch.repeat_interleave(critical_edges, critical_edge_repeat, dim=1)

            # grow critical path by one edge
            critical_paths = torch.cat([rearranged_critical_paths, rearranged_critical_edges[[1], :]], dim=0)

            # collect each critical path to corresponding instance
            path_end_mask = torch.isin(critical_paths[-1], T)
            if path_end_mask.sum() != 0:
                path = critical_paths[:, path_end_mask]
                for p_id in range(path.shape[1]):
                    p = path[:, p_id]
                    instance_id = torch.eq(p[-1], T).nonzero()[0]
                    collect_path[instance_id].append(p)

        return collect_path

    def reset(self,
              instances,
              init_sol_type,
              tabu_size,
              device,
              mask_previous_action=False,
              p_lists=None,
              longest_path_finder='pytorch'):
        self.itr = 0
        self.instances = instances
        self.instance_size = np.array([ins.shape[1:] for ins in instances])
        self.job_count = torch.tensor([instance.shape[1] for instance in instances], dtype=torch.int, device=device)
        self.machine_count = torch.tensor([instance.shape[2] for instance in instances], dtype=torch.int, device=device)
        self.size = torch.stack([self.job_count, self.machine_count]).t()
        self.num_nodes_per_example = self.job_count * self.machine_count + 2
        self.S = (torch.cumsum(self.num_nodes_per_example, dim=0) - self.num_nodes_per_example).cpu().numpy()
        self.T = (torch.cumsum(self.num_nodes_per_example, dim=0) - 1).cpu().numpy()
        self.num_instance = len(instances)
        self.tabu_list = [-torch.ones(size=[tabu_size, 2], device=device, dtype=torch.int64) for _ in
                          range(self.num_instance)]
        self.tabu_size = tabu_size
        self.previous_action = [torch.tensor([-1, -1], device=device, dtype=torch.int64) for _ in
                                range(self.num_instance)]
        # do not consider backward move if True, e.g., [5, 8] will be excluded if previous move is [8, 5], default True.
        self.mask_previous_action = mask_previous_action
        self.longest_path_finder = longest_path_finder
        self._machine_count_cumsum = torch.repeat_interleave(
            self.machine_count.cumsum(dim=0) - self.machine_count,
            self.num_nodes_per_example)

        # print('Computing initial solutions...')
        G_batch, make_span, count = self._init_solver(init=init_sol_type, device=device, p_lists=p_lists)

        self.current_objs, self.incumbent_objs = make_span, make_span
        self.G_batch = G_batch

        return self.G_batch, self.get_candidate_moves()


if __name__ == '__main__':
    from generateJSP import uni_instance_gen

    # j, m, batch_size = {'low': 100, 'high': 101}, {'low': 20, 'high': 21}, 500
    # j, m, batch_size = {'low': 30, 'high': 31}, {'low': 20, 'high': 21}, 128
    j, m, batch_size = {'low': 3, 'high': 6}, {'low': 3, 'high': 6}, 2
    l = 1
    h = 99
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_type = 'fdd-divide-wkr'  # 'spt', 'fdd-divide-wkr', 'plist'
    seed = 1  # 6: two paths for the second instance
    np.random.seed(seed)
    mask_previous = False  # if true usually do not pass N5 property
    print_step_time = True
    print_action_space_compute_time = True
    path_finder = 'pytorch'  # 'networkx' or 'pytorch'
    tb_size = 20  # tabu_size
    torch.random.manual_seed(seed)
    random.seed(seed)

    insts = [np.concatenate(
        [uni_instance_gen(n_j=np.random.randint(**j), n_m=np.random.randint(**m), low=l, high=h)]
    ) for _ in range(batch_size)]

    # insts = np.load('../test_data/tai20x15.npy')
    # print(insts)

    env = Env()
    env.reset(
        instances=insts,
        init_sol_type=init_type,
        tabu_size=tb_size,
        device=dev,
        mask_previous_action=mask_previous,
        longest_path_finder=path_finder
    )
    # print(env.current_objs)
    # print(env.num_nodes_per_example - 2)

    t1 = time.time()
    for _ in range(30):
        print('step {}'.format(_))
        env.step(prt=print_step_time, show_action_space_compute_time=print_action_space_compute_time)
        # print(env.current_objs)
        # print(env.incumbent_objs)
        print()
    t2 = time.time()
    # print(t2 - t1)
    env.cpm_eval()

    # print(env.get_candidate_moves(prt=False)[0])

    # import cProfile
    # cProfile.run('env.longest_paths()', filename='../restats')

    ttt = time.time()
    collected_paths = env.longest_paths()
    print(time.time() - ttt)

    G = env.G_batch
    G.edge_attr = -env.G_batch.dur[env.G_batch.edge_index[0]]
    nx_g = torch_geometric.utils.to_networkx(G, edge_attrs=['edge_attr'], remove_self_loops=True)
    tt = time.time()
    for i, (s, t) in enumerate(zip(env.S, env.T)):
        paths = list(nx.all_shortest_paths(nx_g, source=s, target=t, weight='edge_attr', method='bellman-ford'))
        path0 = paths[0]  # always select the first path
        neural_path = [p.cpu().numpy().tolist() for p in collected_paths[i]]
        if sorted(neural_path) != sorted(paths):
            print('NO!!!!!!!!!!!!!!!!!!!!!!')
            print(sorted(neural_path))
            print(sorted(paths))
    print(time.time() - tt)

    # validate with CPM
    env.cpm_eval()

    # validate N5 property
    from message_passing_evl import MinimalJobshopSat

    ortools_makespan = []
    pygs = []
    eva = MassagePassingEval()
    for _, inst in enumerate(insts):
        print('Processing instance:', _ + 1)
        times_rearrange = np.expand_dims(inst[0], axis=-1)
        machines_rearrange = np.expand_dims(inst[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        val, sol = MinimalJobshopSat(data.tolist())
        ortools_makespan.append(val[1])
    ortools_makespan = np.array(ortools_makespan, dtype=float)
    # which instance got optimal solution
    indeed_optimal = np.nonzero(env.current_objs.squeeze().cpu().numpy() == ortools_makespan)[0]
    # which instance got empty N5 neighbourhood
    should_be_optimal = np.nonzero(np.array([len(m) for m in env.get_candidate_moves()[0]]) == 0)[0]
    print(indeed_optimal)
    print(should_be_optimal)
    if np.isin(should_be_optimal, indeed_optimal).sum() == should_be_optimal.shape[0]:
        print("Pass N5 property check (property in page 6 of the paper).")
    else:
        print("!!!Do not pass N5 property check (property in page 6 of the paper).")
