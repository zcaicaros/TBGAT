import time
import random
import numpy as np
import numpy.random
import torch
import torch.optim as optim
from env.environment import Env
from model.actor import Actor
from env.generateJSP import uni_instance_gen as inst_gen
from pathlib import Path
from env.message_passing_evl import exact_solver
from parameters import args
from tqdm import tqdm


class NeuralTabu:
    def __init__(self):

        self.env_training = Env()
        self.env_validation = Env()
        self.eps = np.finfo(np.float32).eps.item()
        self.algo_config = '{}_{}-{}-{}-{}_{}x{}-{}-{}-{}-{}-{}-{}-{}'.format(
            # env parameters
            args.tabu_size,
            # model parameters
            args.hidden_channels, args.out_channels, args.heads, args.dropout_for_gat,
            # training parameters
            args.j, args.m, args.lr, args.steps_learn, args.transit, args.batch_size, args.total_instances,
            args.step_validation, args.ent_coeff
        )

        # load or generate validation dataset
        validation_data_path = './validation_data/validation_data_and_Cmax_{}x{}_[{},{}].npy'.format(
            args.j, args.m, args.l, args.h
        )
        # load
        if Path(validation_data_path).is_file():
            self.validation_data_and_Cmax = np.load(validation_data_path)  # [#instance, 3, j, m] last of dim=1 is Cmax
            self.validation_data = self.validation_data_and_Cmax[:, [0, 1], :, :]
            self.validation_Cmax = self.validation_data_and_Cmax[:, -1, 0, 0].astype(float)
        # generate
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array(
                [inst_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(args.validation_inst_number)]
            )

            print('Solving validation data using CP-SAT with 3600s limit...')
            self.validation_Cmax, _ = exact_solver(self.validation_data)
            # save validation data and Cmax
            np.save(
                validation_data_path,
                np.concatenate([
                    self.validation_data,
                    self.validation_Cmax.reshape(
                        -1, 1, 1, 1
                    ).repeat(
                        axis=2, repeats=self.validation_data.shape[2]
                    ).repeat(
                        axis=3, repeats=self.validation_data.shape[3]
                    )
                ], axis=1).astype(int)
            )

        # for log and save model
        self.gap_incumbent = np.inf
        self.gap_last_step = np.inf

    def learn(self, rewards, log_probs, ents, optimizer):

        # compute discounted return
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)

        # normalizing return
        normalized_return = torch.div(
            returns - returns.mean(dim=-1, keepdim=True),
            torch.std(returns, dim=-1, unbiased=False, keepdim=True) + self.eps
        )

        # compute log p
        log_probs = torch.cat(log_probs, dim=-1)

        # compute entropy loss
        ents = torch.cat(ents, dim=-1)

        # compute REINFORCE loss with entropy loss
        loss = - (log_probs * normalized_return + args.ent_coeff * ents).sum(dim=-1).mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def validation(self, policy, dev):

        # reset validation env
        validation_start = time.time()
        G, (action_set, optimal_mark, paths) = self.env_validation.reset(
            instances=self.validation_data,
            init_sol_type=args.init_type,
            tabu_size=args.tabu_size,
            device=dev,
            mask_previous_action=args.mask_previous_action,
            longest_path_finder=args.path_finder
        )

        # rollout
        while self.env_validation.itr < args.transit:
            sampled_a, log_p, ent = policy(
                pyg_sol=G,
                feasible_action=action_set,
                optimal_mark=optimal_mark,
                critical_path=paths
            )

            G, reward, (action_set, optimal_mark, paths) = self.env_validation.step(
                action=sampled_a,
                prt=False,
                show_action_space_compute_time=False
            )

        # calculate optimal gap
        result_incumbent = self.env_validation.incumbent_objs.cpu().numpy()
        result_last_step = self.env_validation.current_objs.cpu().numpy()
        gap_incumbent = ((result_incumbent - self.validation_Cmax) / self.validation_Cmax).mean()
        gap_last_step = ((result_last_step - self.validation_Cmax) / self.validation_Cmax).mean()
        validation_end = time.time()

        # saving model based on validation results
        if gap_incumbent < self.gap_incumbent:
            # print('Find better model w.r.t incumbent objs, saving model...')
            torch.save(
                policy.state_dict(),
                './saved_model/incumbent_model_{}.pth'.format(
                    self.algo_config
                )
            )
            self.gap_incumbent = gap_incumbent

        if gap_last_step < self.gap_last_step:
            self.gap_last_step = gap_last_step

        return gap_incumbent, gap_last_step

    def train(self):

        # training seeds
        torch.manual_seed(args.training_seed)
        random.seed(args.training_seed)
        np.random.seed(args.training_seed)
        torch.cuda.manual_seed_all(args.training_seed)

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        print()
        print("Use {} to train...".format(dev))
        print("{}x{}, "
              "lr={}, "
              "ent_coeff={}, "
              "tabu_size={}, "
              "hidden_channels={}, "
              "out_channels={}, "
              "heads={}, "
              "dropout_for_gat={}, "
              "steps_learn={}, "
              "transit={}, "
              "batch_size={}, "
              "total_instances={}, "
              "step_validation={}.".format(args.j,
                                            args.m,
                                            args.lr,
                                            args.ent_coeff,
                                            args.tabu_size,
                                            args.hidden_channels,
                                            args.out_channels,
                                            args.heads,
                                            args.dropout_for_gat,
                                            args.steps_learn,
                                            args.transit,
                                            args.batch_size,
                                            args.total_instances,
                                            args.step_validation
                                            )
              )
        print()

        policy = Actor(
            in_channels_fwd=args.in_channels_fwd,
            in_channels_bwd=args.in_channels_bwd,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            heads=args.heads,
            dropout_for_gat=args.dropout_for_gat
        ).to(dev)

        optimizer = optim.Adam(policy.parameters(), lr=args.lr)

        training_log = []
        validation_log = []

        # add progress bar
        pbar = tqdm(range(1, args.total_instances // args.batch_size + 1))

        for batch_i in pbar:

            t1 = time.time()

            # generate training data on the fly
            instances = np.array([inst_gen(args.j, args.m, args.l, args.h) for _ in range(args.batch_size)])

            # reset the training env with training data
            G, (action_set, optimal_mark, paths) = self.env_training.reset(
                instances=instances,
                init_sol_type=args.init_type,
                tabu_size=args.tabu_size,
                device=dev,
                mask_previous_action=args.mask_previous_action,
                longest_path_finder=args.path_finder)

            reward_buffer = []
            log_prob_buffer = []
            entropy_buffer = []

            while self.env_training.itr < args.transit:

                # forward
                sampled_a, log_p, ent = policy(
                    pyg_sol=G,
                    feasible_action=action_set,
                    optimal_mark=optimal_mark,
                    critical_path=paths
                )

                # step
                G, reward, (action_set, optimal_mark, paths) = self.env_training.step(
                    action=sampled_a,
                    prt=False,
                    show_action_space_compute_time=False
                )

                # store training data
                reward_buffer.append(reward)
                log_prob_buffer.append(log_p)
                entropy_buffer.append(ent)

                if self.env_training.itr % args.steps_learn == 0:
                    # training...
                    self.learn(reward_buffer, log_prob_buffer, entropy_buffer, optimizer)
                    # clean training data
                    reward_buffer = []
                    log_prob_buffer = []
                    entropy_buffer = []

            t2 = time.time()

            # training log
            training_log.append(self.env_training.current_objs.mean().cpu().item())

            pbar.set_postfix(
                {'Batch Mean Performance': '{:.2f} ({:.2f}s)'.format(
                    self.env_training.current_objs.cpu().mean().item(),
                    t2 - t1
                ),
                    'V-IC': '{:.4f}'.format(self.gap_incumbent),
                    'V-LS': '{:.4f}'.format(self.gap_last_step)}
            )

            # start validation and saving model & logs...
            if batch_i % args.step_validation == 0:
                gap_incumbent, gap_last_step = self.validation(policy, dev)
                validation_log.append([gap_incumbent, gap_last_step])
                np.save('./log/validation_log_{}.npy'.format(self.algo_config), np.array(validation_log))
                np.save('./log/training_log_{}.npy'.format(self.algo_config), np.array(training_log))


if __name__ == '__main__':
    agent = NeuralTabu()
    agent.train()
