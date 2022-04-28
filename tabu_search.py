import torch_geometric.utils

from env.message_passing_evl import MassagePassingEval
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
        # one-step forward env init
        self.env_one_step = Env()

    def solve(self):

        # reset rollout env with testing instances
        G, (action_set, _, _) = self.env_rollout.reset(
            instances=self.instances,
            init_sol_type='fdd-divide-wkr',
            tabu_size=self.tabu_size,
            device=self.device,
            mask_previous_action=False,
            longest_path_finder='pytorch')

        while self.env_rollout.itr < 2:
            selected_action = self.calculate_move(G, action_set, self.env_rollout.current_objs)
            G, _, (action_set, _, _) = self.env_rollout.step(
                action=selected_action,
                prt=False,
                show_action_space_compute_time=False
            )

    def calculate_move(self, current_G, current_action_set, current_Cmax):
        ## Select move
        # copy G w.r.t. number of actions
        G_list = current_G.to_data_list()
        print(len(G_list))
        G_expanded = []
        repeats = []

        for _, (a, g) in enumerate(zip(current_action_set, G_list)):
            if not a:
                G_expanded += [g.clone()]
                repeats += [1]
            else:
                G_expanded += [g.clone() for _ in range(a[0].shape[0])]
                repeats += [a[0].shape[0]]
            print(self.env_rollout.num_nodes_per_example)
            print(self.env_rollout.num_nodes_per_example[[_]])
            self.evaluator.eval(
                g,
                num_nodes_per_example=self.env_rollout.num_nodes_per_example[[_]]
            )
        G_expanded = Batch.from_data_list(G_expanded)
        num_nodes_per_example_expanded = torch.repeat_interleave(
            self.env_rollout.num_nodes_per_example,
            repeats=torch.tensor(repeats, device=self.device)
        )

        print(current_action_set)
        print(num_nodes_per_example_expanded)

        est, lst, make_span, _, _, _ = self.evaluator.eval(
            G_expanded,
            num_nodes_per_example=num_nodes_per_example_expanded
        )

        G_expanded.est = est
        G_expanded.lst = lst

        # Reset one-step env
        self.env_one_step.num_nodes_per_example = num_nodes_per_example_expanded
        self.env_one_step.G_batch = G_expanded
        self.env_one_step.S = (
                    torch.cumsum(num_nodes_per_example_expanded, dim=0) - num_nodes_per_example_expanded).cpu().numpy()
        self.env_one_step.T = (torch.cumsum(num_nodes_per_example_expanded, dim=0) - 1).cpu().numpy()
        self.env_one_step.num_instance = self.env_one_step.S.shape[0]
        self.env_one_step.incumbent_objs = torch.repeat_interleave(current_Cmax, repeats=torch.tensor(repeats, device=self.device))
        self.env_one_step.machine_count = torch.repeat_interleave(
            torch.tensor(
                [instance.shape[2] for instance in self.instances],
                dtype=torch.int,
                device=self.device
            ), repeats=torch.tensor(repeats, device=self.device))
        self.env_one_step._machine_count_cumsum = torch.repeat_interleave(
            self.env_one_step.machine_count.cumsum(dim=0) - self.env_one_step.machine_count,
            self.env_one_step.num_nodes_per_example
        )
        self.env_one_step.tabu_list = [
            -torch.ones(size=[self.tabu_size, 2], device=self.device, dtype=torch.int64)
            for _ in range(self.env_one_step.num_instance)
        ]
        self.env_one_step.previous_action = [
            torch.tensor([-1, -1], device=self.device, dtype=torch.int64) for _ in
            range(self.env_one_step.num_instance)]

        # prepare actions
        _operation_index_helper1 = torch.cumsum(
            self.env_one_step.num_nodes_per_example, dim=0
        ) - self.env_one_step.num_nodes_per_example

        _operation_index_helper2 = torch.cumsum(
            self.env_rollout.num_nodes_per_example, dim=0
        ) - self.env_rollout.num_nodes_per_example

        print(_operation_index_helper1)
        print(_operation_index_helper2)
        print(torch.cat(
            [actions[0][:, :2] - _operation_index_helper2[_] for _, actions in enumerate(current_action_set) if actions], dim=0
        ))

        action_merged_one_step = torch.cat(
            [actions[0][:, :2] - _operation_index_helper2[_] for _, actions in enumerate(current_action_set) if actions], dim=0
        ) + _operation_index_helper1.unsqueeze(-1)

        # step one-step forward to test all candidate actions
        _Cmax_before_step = self.env_one_step.incumbent_objs
        # print(_Cmax_before_step)
        current_G, reward, (_, _, _) = self.env_one_step.step(
            action=action_merged_one_step,
            prt=False,
            show_action_space_compute_time=False
        )
        _Cmax_after_step = self.env_one_step.incumbent_objs
        # print(_Cmax_after_step)
        Cmax_decrease_label = (_Cmax_after_step < _Cmax_before_step)
        # print(Cmax_decrease_label)
        tabu_label = torch.cat([actions[0] for actions in current_action_set if actions], dim=0)[:, -1].bool()  # True: tabu
        # print(tabu_label)
        # print(current_action_set)

        selected_action = torch.cat([actions[0][[0], :2] for actions in current_action_set if actions], dim=0)
        print(selected_action)

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
