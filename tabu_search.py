import torch_geometric.utils

from env.message_passing_evl import MassagePassingEval
from torch_geometric.utils import sort_edge_index
from torch_geometric.data.batch import Batch
from ortools_solver import MinimalJobshopSat
from env.environment import Env
import numpy as np
import random
import torch


class TSN5:
    def __init__(self,
                 instances,
                 search_horizons,
                 tabu_size,
                 device):

        self.instances = instances
        self.search_horizons = search_horizons
        self.tabu_size = tabu_size
        self.device = device
        self.evaluator = MassagePassingEval()
        # rollout env init
        self.env_rollout = Env()

    def solve(self):

        # reset rollout env with testing instances
        G, (action_set, _, _) = self.env_rollout.reset(
            instances=self.instances,
            init_sol_type='fdd-divide-wkr',
            tabu_size=self.tabu_size,
            device=self.device,
            mask_previous_action=False,
            longest_path_finder='pytorch')

        while self.env_rollout.itr < 5:
            selected_action = self.calculate_move(G, self.env_rollout.current_objs, action_set)
            G, _, (action_set, _, _) = self.env_rollout.step(
                action=selected_action,
                prt=False,
                show_action_space_compute_time=False
            )

    def calculate_move(self, current_sol, current_cmax, current_action_set):

        # sort edge_index otherwise to_data_list() fn will be messed and bug
        current_sol.edge_index = sort_edge_index(current_sol.edge_index)
        # sort edge_index_disjunctions otherwise to_data_list() fn will be messed and bug
        current_sol.edge_index_disjunctions = sort_edge_index(current_sol.edge_index_disjunctions)

        # copy G for one-step forward
        G_list = current_sol.to_data_list()
        num_nodes_per_example = torch.tensor([G.num_nodes for G in G_list], device=self.device)
        G_expanded = []
        repeats = []
        action_exist = []
        for _, (a, g) in enumerate(zip(current_action_set, G_list)):
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

        ## prepare actions for one-setp rollout
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
            [actions[0][:, :2] - _operation_index_helper1[_] for _, actions in enumerate(current_action_set) if
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
        Cmax_before_one_step = current_cmax.repeat_interleave(
            repeats=torch.tensor(repeats, device=self.device)
        )

        # Cmax after one step
        G_expanded.edge_index = torch.cat([edge_index_disjunctions, G_expanded.edge_index_conjunctions], dim=1)
        _, _, Cmax_after_one_step, _, _, _ = self.evaluator.eval(
            G_expanded,
            num_nodes_per_example=num_nodes_per_example_one_step
        )

        # Cmax improved flag, True(1) for improved
        Cmax_improved_flag = torch.lt(Cmax_after_one_step, Cmax_before_one_step).int()

        # tabu label, True(1) for tabu
        tabu_label = torch.cat([actions[0] for actions in current_action_set if actions], dim=0)[:, 2]

        # compute aspiration
        aspiration_flag = Cmax_improved_flag[action_exist]



        print(tabu_label)
        print(aspiration_flag)
        print(Cmax_improved_flag)
        print()

        selected_action = torch.cat([actions[0][[0], :2] for actions in current_action_set if actions], dim=0)
        # print(selected_action)

        return selected_action


if __name__ == '__main__':

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    np.seterr(invalid='ignore')

    dev = 'cpu' if torch.cuda.is_available() else 'cpu'
    print('using to test TSN5...'.format(dev))

    # benchmark config
    init_type = ['fdd-divide-wkr']  # ['fdd-divide-wkr', 'spt']
    testing_type = ['syn']  # ['tai', 'abz', 'ft', 'la', 'swv', 'orb', 'yn']
    syn_problem_j = [10]  # [10, 15, 15, 20, 20, 100, 150]
    syn_problem_m = [10]  # [10, 10, 15, 10, 15, 20, 25]
    tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]
    abz_problem_j = [10, 20]
    abz_problem_m = [10, 15]
    orb_problem_j = [10]
    orb_problem_m = [10]
    yn_problem_j = [20]
    yn_problem_m = [20]
    swv_problem_j = [20, 20, 50]
    swv_problem_m = [10, 15, 10]
    la_problem_j = [10, 15, 20, 10, 15, 20, 30, 15]  # [10, 15, 20, 10, 15, 20, 30, 15]
    la_problem_m = [5, 5, 5, 10, 10, 10, 10, 15]  # [5, 5, 5, 10, 10, 10, 10, 15]
    ft_problem_j = [6, 10, 20]  # [6, 10, 20]
    ft_problem_m = [6, 10, 5]  # [6, 10, 5]

    # solver config
    taboo_size = 20
    cap_horizon = 5000
    performance_milestones = [1, 2, 3, 4]  # [500, 1000, 2000, 5000]
    result_type = 'incumbent'  # 'current', 'incumbent'

    for test_t in testing_type:  # select benchmark
        if test_t == 'syn':
            problem_j, problem_m = syn_problem_j, syn_problem_m
        elif test_t == 'tai':
            problem_j, problem_m = tai_problem_j, tai_problem_m
        elif test_t == 'abz':
            problem_j, problem_m = abz_problem_j, abz_problem_m
        elif test_t == 'orb':
            problem_j, problem_m = orb_problem_j, orb_problem_m
        elif test_t == 'yn':
            problem_j, problem_m = yn_problem_j, yn_problem_m
        elif test_t == 'swv':
            problem_j, problem_m = swv_problem_j, swv_problem_m
        elif test_t == 'la':
            problem_j, problem_m = la_problem_j, la_problem_m
        elif test_t == 'ft':
            problem_j, problem_m = ft_problem_j, ft_problem_m
        else:
            raise Exception(
                'Problem type must be in testing_type = ["tai", "abz", "orb", "yn", "swv", "la", "ft", "syn"].')

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size

            # inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))[:2]

            from env.generateJSP import uni_instance_gen
            j, m, l, h, batch_size = {'low': 3, 'high': 6}, {'low': 3, 'high': 6}, 1, 99, 3
            inst = [np.concatenate(
                [uni_instance_gen(n_j=np.random.randint(**j), n_m=np.random.randint(**m), low=l, high=h)]
            ) for _ in range(batch_size)]

            print('\nStart testing {}{}x{}...'.format(test_t, p_j, p_m))

            # read saved gap_against or use ortools to solve it.
            if test_t != 'syn':
                gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
            else:
                # ortools solver
                from pathlib import Path

                ortools_path = Path('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                if ortools_path.is_file():
                    gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                else:
                    ortools_results = []
                    print('Starting Ortools...')
                    for i, data in enumerate(inst):
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                        ortools_results.append(result)
                    ortools_results = np.array(ortools_results)
                    np.save('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m), ortools_results)
                    gap_against = ortools_results[:, 1]

            # start to test
            solver = TSN5(instances=inst, search_horizons=performance_milestones, tabu_size=taboo_size, device=dev)
            solver.solve()
