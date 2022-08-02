import numpy as np
import torch
import time
import random
from env.environment import Env
from model.actor import Actor
from ortools_solver import MinimalJobshopSat
from parameters import args
import pandas as pd


def main():
    seed = args.t_seed
    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide='ignore')  # rm RuntimeWarning: divide by zero encountered in true_divide priority = fdd/wkr of Orb

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} to test...'.format(dev))

    # MDP config
    performance_milestones = [500, 1000, 2000, 5000]  # [500, 1000, 2000, 5000]
    result_type = 'incumbent'  # 'last_step', 'incumbent'
    init = 'fdd-divide-wkr'  # 'fdd-divide-wkr', 'spt'

    # testing specific size
    if args.test_specific_size == 'True':

        # which model to load
        algo_config = '{}-{}_{}-{}-{}-{}-{}_FSSP-{}x{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            # env parameters
            args.tabu_size,
            args.init_type,
            # model parameters
            args.hidden_channels, args.out_channels, args.heads, args.dropout_for_gat, args.embed_net,
            # training parameters
            args.j, args.m, args.lr, args.steps_learn, args.transit, args.batch_size, args.total_instances,
            args.step_validation, args.ent_coeff, args.training_seed, args.embed_tabu_label, args.action_selection_type
        )

        test_instance_size = [p_j, p_m] = [args.t_j, args.t_m]
        print('Testing all open benchmark of size {}.'.format(test_instance_size))
        if test_instance_size == [20, 5]:
            testing_type = ['tai']
        elif test_instance_size == [20, 10]:
            testing_type = ['tai']
        elif test_instance_size == [20, 20]:
            testing_type = ['tai']
        else:
            raise RuntimeError('Open benchmark has no instances of size: {}.'.format(test_instance_size))

        for test_t in testing_type:  # select benchmark

            inst = np.load('./test_data_fssp/{}{}x{}.npy'.format(test_t, p_j, p_m))  # [[0], :, :, :]

            print('\nStart testing FSSP-{}{}x{}...'.format(test_t, p_j, p_m))

            # read saved gap_against or use ortools to solve it.
            gap_against = np.load('./test_data_fssp/{}{}x{}_result.npy'.format(test_t, p_j, p_m))

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

            pytorch_total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            print('Total number of parameters of model: \n{}\n is:'.format(saved_model_path), pytorch_total_params)

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print('Starting rollout DRL policy...')
            # t3 = time.time()
            result, computation_time = [], []
            G, (action_set, optimal_mark, paths) = env.reset(
                instances=inst,
                init_sol_type=init,
                tabu_size=args.tabu_size,
                device=dev,
                mask_previous_action=args.mask_previous_action == 'True',
                longest_path_finder=args.path_finder)
            # t4 = time.time()
            drl_start = time.time()
            while env.itr < max(performance_milestones):
                # t1 = time.time()
                sampled_a, log_p, ent = policy(
                    pyg_sol=G,
                    feasible_action=action_set,
                    optimal_mark=optimal_mark
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
                        print(
                            'For testing steps: {}    '.format(
                                env.itr if env.itr > min(performance_milestones) else ' ' + str(env.itr)),
                            'Optimal Gap: {:.6f}    '.format(((DRL_result - gap_against) / gap_against).mean()),
                            'Average Time: {:.4f}    '.format(computation_time[-1] / inst.shape[0]),
                            "Cmax is: {}".format(env.incumbent_objs.cpu().numpy()  # show makespan explicitly
                                                 )
                        )
    # testing all benchmark
    else:
        print('Testing all instances of all sizes using all models.')

        # should be manually set to the size of model you have trained, see 'saved_model' folder.
        model_size = [
            [10, 10],
            [20, 5],
            [20, 10],
            [20, 20]
        ]

        # model_size = [
        #     [6, 6],
        # ]

        mean_gap_all_model_all_benchmark = []
        mean_time_all_model_all_benchmark = []
        csv_index = []

        env_model_config = '{}-{}_{}-{}-{}-{}-{}'.format(
            # env parameters
            args.tabu_size,
            args.init_type,
            # model parameters
            args.hidden_channels, args.out_channels, args.heads, args.dropout_for_gat, args.embed_net
        )

        training_config = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            # training parameters
            args.lr, args.steps_learn, args.transit, args.batch_size,
            args.total_instances, args.step_validation, args.ent_coeff, args.training_seed, args.embed_tabu_label,
            args.action_selection_type
        )

        for [model_j, model_m] in model_size:

            testing_type = ['tai']  # ['tai']
            tai_problem_j = [20, 20, 20]  # [20, 20, 20]
            tai_problem_m = [5, 10, 20]  # [5, 15, 20]

            mean_gap_all_benchmark = []
            mean_time_all_benchmark = []

            for test_t in testing_type:  # select benchmark
                if test_t == 'tai':
                    problem_j, problem_m = tai_problem_j, tai_problem_m
                else:
                    raise Exception(
                        'Problem type must be in testing_type = ["tai"].')

                mean_gap_each_bench = []
                mean_time_each_bench = []

                for p_j, p_m in zip(problem_j, problem_m):  # select problem size

                    inst = np.load('./test_data_fssp/{}{}x{}.npy'.format(test_t, p_j, p_m))

                    print('\nStart testing FSSP-{}{}x{}...'.format(test_t, p_j, p_m))

                    # read saved gap_against or use ortools to solve it.
                    gap_against = np.load('./test_data_fssp/{}{}x{}_result.npy'.format(test_t, p_j, p_m))

                    env = Env()
                    policy = Actor(
                        in_channels_fwd=args.in_channels_fwd,
                        in_channels_bwd=args.in_channels_bwd,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.out_channels,
                        heads=args.heads,
                        dropout_for_gat=args.dropout_for_gat
                    ).to(dev).eval()

                    model_size_config = 'FSSP-{}x{}'.format(model_j, model_m)

                    algo_config = env_model_config + '_' + model_size_config + '-' + training_config

                    saved_model_path = './saved_model/incumbent_model_' + algo_config + '.pth'
                    print('loading model from:', saved_model_path)
                    policy.load_state_dict(torch.load(saved_model_path, map_location=torch.device(dev)))

                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    print('Starting rollout DRL policy...')
                    # t3 = time.time()
                    G, (action_set, optimal_mark, paths) = env.reset(
                        instances=inst,
                        init_sol_type=init,
                        tabu_size=args.tabu_size,
                        device=dev,
                        mask_previous_action=args.mask_previous_action == 'True',
                        longest_path_finder=args.path_finder)
                    # t4 = time.time()

                    mean_gap_each_size = []
                    mean_time_each_size = []

                    drl_start = time.time()
                    while env.itr < max(performance_milestones):
                        # t1 = time.time()
                        sampled_a, log_p, ent = policy(
                            pyg_sol=G,
                            feasible_action=action_set,
                            optimal_mark=optimal_mark
                        )

                        G, reward, (action_set, optimal_mark, paths) = env.step(
                            action=sampled_a,
                            prt=False,
                            show_action_space_compute_time=False
                        )

                        # t2 = time.time()
                        for log_horizon in performance_milestones:
                            if env.itr == log_horizon:
                                time_milestone = time.time() - drl_start
                                csv_index.append('{} {}x{} {}'.format(test_t, p_j, p_m, log_horizon))
                                if result_type == 'incumbent':
                                    DRL_result = env.incumbent_objs.cpu().squeeze().numpy()
                                else:
                                    DRL_result = env.current_objs.cpu().squeeze().numpy()
                                print('For testing steps: {}    '.format(
                                    env.itr if env.itr > 500 else ' ' + str(env.itr)),
                                    'Optimal Gap: {:.6f}    '.format(
                                        ((DRL_result - gap_against) / gap_against).mean()),
                                    'Average Time: {:.4f}    '.format(time_milestone / inst.shape[0]),
                                    "Cmax is: {}".format(env.incumbent_objs.cpu().numpy()  # show makespan explicitly
                                                         )
                                )
                                # show makespan explicitly
                                # print()
                                mean_gap_each_size.append(((DRL_result - gap_against) / gap_against).mean())
                                mean_time_each_size.append(time_milestone / inst.shape[0])
                    mean_time_each_bench.append(np.array(mean_time_each_size).reshape(-1, 1))
                    mean_gap_each_bench.append(np.array(mean_gap_each_size).reshape(-1, 1))
                mean_gap_all_benchmark.append(np.concatenate(mean_gap_each_bench, axis=0))
                mean_time_all_benchmark.append(np.concatenate(mean_time_each_bench, axis=0))
                mean_gap_all_benchmark.append(np.array([[-1]], dtype=float))
                mean_time_all_benchmark.append(np.array([[-1]], dtype=float))
                csv_index.append('dummy')
            mean_gap_all_benchmark = np.concatenate(mean_gap_all_benchmark, axis=0)
            mean_time_all_benchmark = np.concatenate(mean_time_all_benchmark, axis=0)
            mean_gap_all_model_all_benchmark.append(mean_gap_all_benchmark)
            mean_time_all_model_all_benchmark.append(mean_time_all_benchmark)
        mean_gap_all_model_all_benchmark = np.concatenate(mean_gap_all_model_all_benchmark, axis=1)
        mean_time_all_model_all_benchmark = np.concatenate(mean_time_all_model_all_benchmark, axis=1)
        dataFrame_gap = pd.DataFrame(
            mean_gap_all_model_all_benchmark,
            index=csv_index[:mean_gap_all_model_all_benchmark.shape[0]],
            columns=['{}x{}'.format(model_j, model_m) for [model_j, model_m] in model_size])
        dataFrame_time = pd.DataFrame(
            mean_time_all_model_all_benchmark,
            index=csv_index[:mean_time_all_model_all_benchmark.shape[0]],
            columns=['{}x{}'.format(model_j, model_m) for [model_j, model_m] in model_size])
        # writing to excel
        with pd.ExcelWriter('excel/FSSP-{}.xlsx'.format(env_model_config + '_' + training_config)) as writer:
            dataFrame_gap.to_excel(
                writer,
                sheet_name='mean gap',  # sheet name
                float_format='%.8f'
            )
            dataFrame_time.to_excel(
                writer,
                sheet_name='mean time',  # sheet name
                float_format='%.8f'
            )


if __name__ == '__main__':
    main()
