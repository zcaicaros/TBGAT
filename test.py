import numpy as np
import torch
import time
import random
from env.environment import Env
from model.actor import Actor
from ortools_solver import MinimalJobshopSat
from parameters import args


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide='ignore')  # rm RuntimeWarning: divide by zero encountered in true_divide priority = fdd/wkr of Orb

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} to test...'.format(dev))

    # env parameters
    tabu_size = 20
    # model parameters
    hidden_channels = 128
    out_channels = 128
    heads = 8
    dropout_for_gat = 0
    # training parameters
    j = 10
    m = 10
    lr = 1e-5
    steps_learn = 10
    transit = 500
    batch_size = 64
    total_instances = 32000
    step_validation = 10
    ent_coeff = 1e-5

    algo_config = '{}_{}-{}-{}-{}_{}x{}-{}-{}-{}-{}-{}-{}-{}'.format(
        # env parameters
        tabu_size,
        # model parameters
        hidden_channels, out_channels, heads, dropout_for_gat,
        # training parameters
        j, m, lr, steps_learn, transit, batch_size, total_instances, step_validation, ent_coeff
    )

    # benchmark config
    init_type = ['fdd-divide-wkr']  # ['fdd-divide-wkr', 'spt']
    testing_type = ['la']  # ['tai', 'abz', 'ft', 'la', 'swv', 'orb', 'yn']
    # ['abz', 'ft', 'la', 'orb', 'syn'] 10x10
    # ['tai', 'la', 'syn'] 15x15
    # ['tai', 'abz', 'swv', 'syn'] 20x15
    # ['tai', 'yn', 'syn'] 20x20
    # ['tai', 'syn'] 30x15
    # ['tai', 'syn'] 30x20

    # tai_problem_j = [15]  # [15, 20, 20, 30, 30, 50, 50, 100]
    # tai_problem_m = [15]  # [15, 15, 20, 15, 20, 15, 20, 20]
    # abz_problem_j = [10]  # [10, 20]
    # abz_problem_m = [10]  # [10, 15]
    # ft_problem_j = [10]  # [6, 10, 20]
    # ft_problem_m = [10]  # [6, 10, 5]
    # la_problem_j = [10]  # [10, 15, 20, 10, 15, 20, 30, 15]
    # la_problem_m = [10]  # [5, 5, 5, 10, 10, 10, 10, 15]
    # swv_problem_j = [20, 20, 50]  # [20, 20, 50]
    # swv_problem_m = [10, 15, 10]  # [10, 15, 10]
    # orb_problem_j = [10]
    # orb_problem_m = [10]
    # yn_problem_j = [20]
    # yn_problem_m = [20]
    # syn_problem_j = [15]  # [10, 15, 15, 20, 20, 100, 150]
    # syn_problem_m = [15]  # [10, 10, 15, 10, 15, 20, 25]

    tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]  # [15, 20, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]  # [15, 15, 20, 15, 20, 15, 20, 20]
    abz_problem_j = [10, 20]  # [10, 20]
    abz_problem_m = [10, 15]  # [10, 15]
    ft_problem_j = [6, 10, 20]  # [6, 10, 20]
    ft_problem_m = [6, 10, 5]  # [6, 10, 5]
    la_problem_j = [10, 15, 20, 10, 15, 20, 30, 15]  # [10, 15, 20, 10, 15, 20, 30, 15]
    la_problem_m = [5, 5, 5, 10, 10, 10, 10, 15]  # [5, 5, 5, 10, 10, 10, 10, 15]
    swv_problem_j = [20, 20, 50]  # [20, 20, 50]
    swv_problem_m = [10, 15, 10]  # [10, 15, 10]
    orb_problem_j = [10]
    orb_problem_m = [10]
    yn_problem_j = [20]
    yn_problem_m = [20]
    syn_problem_j = [10, 15, 15, 20, 20, 100, 150]  # [10, 15, 15, 20, 20, 100, 150]
    syn_problem_m = [10, 10, 15, 10, 15, 20, 25]  # [10, 10, 15, 10, 15, 20, 25]

    # MDP config
    cap_horizon = 5000
    performance_milestones = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000]
    result_type = 'incumbent'  # 'last_step', 'incumbent'

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

            inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))

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

            env = Env()
            policy = Actor(
                in_channels_fwd=args.in_channels_fwd,
                in_channels_bwd=args.in_channels_bwd,
                hidden_channels=args.hidden_channels,
                out_channels=args.out_channels,
                heads=args.heads,
                dropout_for_gat=args.dropout_for_gat
            ).to(dev).eval()

            saved_model_path = './saved_model/incumbent_model_' + algo_config + '.pth'
            print('loading model from:', saved_model_path)
            policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

            for init in init_type:
                torch.manual_seed(seed)
                print('Starting rollout DRL policy...')
                results_each_init, inference_time_each_init = [], []
                # t3 = time.time()
                result, computation_time = [], []
                G, (action_set, optimal_mark, paths) = env.reset(
                    instances=inst,
                    init_sol_type=init,
                    tabu_size=args.tabu_size,
                    device=dev,
                    mask_previous_action=args.mask_previous_action,
                    longest_path_finder=args.path_finder)
                # t4 = time.time()
                drl_start = time.time()
                while env.itr < cap_horizon:
                    # t1 = time.time()
                    sampled_a, log_p, ent = policy(
                        pyg_sol=G,
                        feasible_action=action_set,
                        optimal_mark=optimal_mark,
                        critical_path=paths
                    )

                    G, reward, (action_set, optimal_mark, paths) = env.step(
                        action=sampled_a,
                        prt=False,
                        show_action_space_compute_time=False
                    )

                    # t2 = time.time()
                    for log_horizon in performance_milestones:
                        if env.itr == log_horizon:
                            if result_type == 'incumbent':
                                DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                            else:
                                DRL_result = env.current_objs.cpu().squeeze().numpy()
                            result.append(DRL_result)
                            computation_time.append(time.time() - drl_start)
                            print('For testing steps: {}    '.format(env.itr if env.itr > 500 else ' ' + str(env.itr)),
                                  'Optimal Gap: {:.6f}    '.format(((DRL_result - gap_against) / gap_against).mean()),
                                  'Average Time: {:.4f}    '.format(computation_time[-1]/inst.shape[0]))
                results_each_init.append(np.stack(result))
                inference_time_each_init.append(np.array(computation_time))


if __name__ == '__main__':

    main()
