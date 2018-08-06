# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from elder_chess_native import Board, random_seed
from elder_chess_game_server import ElderChessGameServer
from mcts_player import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from tensorflow_policy import get_local_server, PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras

import multiprocessing

current_model_path = "models/current_model.model"
best_model_path = "models/best_policy.model"

multiprocessing.set_start_method("spawn", force=True)

class PolicyEvaluator():
    
    def __init__(self, job_queue, out_queue, c_puct, n_playout, current_model_path, best_model_path):
        self.board = Board()
        self.game = ElderChessGameServer(self.board)

        self.server = get_local_server()
        self.policy_value_net = PolicyValueNet(server=self.server)
        self.duplicate_policy_value_net = PolicyValueNet(server=self.server)

        self.job_queue = job_queue
        self.out_queue = out_queue

        self.c_puct = c_puct
        self.n_playout = n_playout
        self.current_model_path = current_model_path
        self.best_model_path = best_model_path

        self.name = multiprocessing.current_process()

    def start_listen(self):
        while True:
            print(self.name, "start listen", flush=True)
            job_type, job_args = self.job_queue.get(True)
            print(self.name, "got", (job_type, job_args), flush=True)
            if job_type == "eval":
                start_i, end_i = job_args
                self.out_queue.put(self.policy_evaluate(start_i, end_i), False)
            if job_type == "done":
                print(self.name, "is done", flush=True)
                return

    def policy_evaluate(self, start_i, end_i):
        self.policy_value_net.restore_model(self.current_model_path)
        self.duplicate_policy_value_net.restore_model(self.best_model_path)

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout, name="Current")
        old_mcts_player = MCTSPlayer(self.duplicate_policy_value_net.policy_value,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout, name="Old")

        win_cnt = [0] * 3
        for i in range(start_i, end_i):
            if i % 2 == 0:
                p1, p2 = current_mcts_player, old_mcts_player
            else:
                p1, p2 = old_mcts_player, current_mcts_player
            winner = self.game.start_play(p1, p2, temp=1., is_shown=False)
            if winner == current_mcts_player:
                win_cnt[0] += 1
            elif winner == old_mcts_player:
                win_cnt[1] += 1
            else:
                win_cnt[2] += 1
            # print("{}: game {} winner is {}".format(self.name, i, winner.name if winner else "tie"))
        win_ratio = 1.0*(win_cnt[0] + 0.5*win_cnt[2]) / (end_i - start_i)
        print("Process {} --- win: {}, lose: {}, tie:{}, win_ratio: {}".format(self.name, win_cnt[0], win_cnt[1], win_cnt[2], win_ratio), flush=True)
        return win_cnt

class TrainPipeline():
    def __init__(self, eval_n_threads=1, init_model=None, current_model_path=current_model_path, best_model_path=best_model_path):
        # params of the board and the game
        self.board_width = 4
        self.board_height = 4
        self.n_in_row = 4
        self.board = Board()
        self.game = ElderChessGameServer(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1500  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 500000
        self.batch_size = 2048  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 5000
        self.win_ratio_cutoff = 0.51
        # misc
        self.eval_n_games = 120
        self.eval_n_threads = eval_n_threads
        self.current_model_path = current_model_path
        self.best_model_path = best_model_path
        self.jobs_launched = False
        if init_model:
            print("starting from model: ", init_model)
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet()
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=True)
    def start_workers(self):
        self.processes = []
        self.job_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        for _ in range(self.eval_n_threads):
            process = multiprocessing.Process(
                target=TrainPipeline.worker_main, 
                args=(self.job_queue, self.out_queue, self.c_puct, self.n_playout, self.current_model_path, self.best_model_path)
            )
            process.start()
            self.processes.append(process)

    def cleanup_workers(self):
        print("cleaning up")
        for _ in range(self.eval_n_threads):
            self.job_queue.put(("done", None))
        for process in self.processes:
            process.join()

    @staticmethod
    def worker_main(in_queue, out_queue, c_puct, n_playout, current_model_path, best_model_path):
        print(multiprocessing.current_process(), "working")
        import time
        seed1 = time.time() + os.getpid() + 42.7
        seed2 = time.time() - os.getpid() + 61.9
        np.random.seed(int(seed1))
        random_seed(seed2) # re-seed each process
        evaluator = PolicyEvaluator(in_queue, out_queue, c_puct, n_playout, current_model_path, best_model_path)
        evaluator.start_listen()

    def rotate_moves(self, moves):
        moves = np.rot90(moves, 1, (1, 2))
        f, u, d, l, r = moves
        return np.array([f, r, l, u, d])

    def flip_moves(self, moves):
        f, u, d, l, r = moves
        return np.array([np.fliplr(f), np.fliplr(u), np.fliplr(d), np.fliplr(r), np.fliplr(l)])

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for (board_state, hidden_pieces, remaining), mcts_prob, winner in play_data:
            remaining = np.array([remaining])
            for _ in range(4):
                # rotate counterclockwise
                board_state = np.array([np.rot90(s, 1) for s in board_state])
                mcts_prob = self.rotate_moves(mcts_prob)
                extend_data.append(((board_state, hidden_pieces, remaining),
                                    mcts_prob.flatten(),
                                    winner))
                # flip horizontally
                board_state_f = np.array([np.fliplr(s) for s in board_state])
                mcts_prob_f = self.flip_moves(mcts_prob)
                extend_data.append(((board_state_f, hidden_pieces, remaining),
                                    mcts_prob_f.flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data_extended = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data_extended)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = (
                [data[0][0] for data in mini_batch], 
                [data[0][1] for data in mini_batch], 
                [data[0][2] for data in mini_batch]
        )
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate_async(self):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        print("launching evaluate jobs")
        self.jobs_launched = True
        games_per_thread = self.eval_n_games // self.eval_n_threads
        res_games = self.eval_n_games % self.eval_n_threads
        games_per_thread = [games_per_thread + 1] * res_games + [games_per_thread] * (self.eval_n_threads - res_games)
        start_i = 0
        for n_games_per_thread in games_per_thread:
            self.job_queue.put(("eval", (start_i, start_i + n_games_per_thread)))
            start_i += n_games_per_thread

    def policy_evaluate_sync(self):
        if not self.jobs_launched:
            return -1
        print("syncing evaluated policies...")
        win_cnt = np.zeros(3)
        for _ in range(self.eval_n_threads):
            wc = self.out_queue.get(True)
            win_cnt += np.array(wc)

        win_ratio = 1.0*(win_cnt[0] + 0.5*win_cnt[2]) / self.eval_n_games
        print("Total --- win: {}, lose: {}, tie:{}".format(win_cnt[0], win_cnt[1], win_cnt[2]))
        self.jobs_launched = False
        return win_ratio

    def run(self):
        """run the training pipeline"""
        self.start_workers()

        try:
            self.policy_value_net.save_model(self.best_model_path)
            self.policy_value_net.save_model(self.current_model_path)
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                if (i+1) % self.check_freq == 0:
                    win_rate = self.policy_evaluate_sync()
                    if win_rate >= self.win_ratio_cutoff:
                        print("New best policy!!!!!!!!")
                        self.policy_value_net.save_model(self.best_model_path)
                    self.policy_value_net.save_model(self.current_model_path)
                    self.policy_evaluate_async()
                    
        except KeyboardInterrupt:
            print('\n\rquit')
        
        self.cleanup_workers()


import os

if __name__ == '__main__':
    if os.path.isfile(best_model_path+".meta"):
        best_model_file = best_model_path
    else:
        best_model_file = None
    training_pipeline = TrainPipeline(init_model=best_model_file)
    training_pipeline.run()
