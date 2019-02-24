from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from .elder_chess_native import Board
from .elder_chess_game_server import ElderChessGameServer
from .mcts_player import MCTSPlayer
from .nn_player import NNPlayer

from .tensorflow_policy import PolicyValueNet

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board = Board()
        self.game = ElderChessGameServer(self.board)
        # training params
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.n_playout = 1000  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 500000
        self.batch_size = 256  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1024
        self.epochs = 5  # num of train_steps for each update
        self.num_updates = 10000
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.win_ratio_cutoff = 0.55
        self.eval_n_games = 20
        if init_model:
            print("starting from model: ", init_model)
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet()
        self.duplicate_policy_value_net = PolicyValueNet()
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=True)
        self.nn_player = NNPlayer(self.policy_value_net.policy_value, is_selfplay=True)
        self.backup_policy()

    def backup_policy(self):
        self.policy_value_net.save_model("models/current_policy.model")
        self.duplicate_policy_value_net.restore_model("models/current_policy.model")

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

        # Collect traces (in parallel)
        boards = [Board() for i in range(n_games)]
        ended_traces = []
        traces = [[] for _ in range(n_games)]
        traces_end_winners = [None for _ in range(n_games)]
        num_ended_games = 0
        while num_ended_games < len(boards):
            state_batch = [b.get_compact_state() for b in boards[num_ended_games:]]
            moves = self.nn_player.get_action_batch(boards[num_ended_games:], state_batch)
            new_boards = []
            cur_num_ended_games = num_ended_games
            for i in range(num_ended_games, len(boards)):
                board, move = boards[i], moves[i - num_ended_games]
                traces[i].append((state_batch[i - num_ended_games], Board(board)))
                if board.do_move_safe(move):
                    if board.is_env_move():
                        board.env_do_move()
                else:
                    raise Exception("Invalid move {} by NNPlayer".format(move))
                if board.game_ended():
                    traces[cur_num_ended_games], traces[i] = traces[i], traces[cur_num_ended_games]
                    boards[cur_num_ended_games], boards[i] = boards[i], boards[cur_num_ended_games]
                    traces_end_winners[cur_num_ended_games] = board.get_winner()
                    cur_num_ended_games += 1
            num_ended_games = cur_num_ended_games
        assert num_ended_games == len(boards)

        play_data = []
        for trace, winner in zip(traces, traces_end_winners):
            for state_batch, board in trace:
                if winner == 2:
                    score = 0.
                elif winner == board.get_current_player():
                    score = 1.0
                else:
                    score = -1.0
                play_data.append((state_batch, board, score))

        del traces

        # Label with expert
        move_and_probs = self.mcts_player.get_action_batch([data[1] for data in play_data], return_prob=True)

        play_data = [(data[0], probs, data[2]) for (m, probs), data in zip(move_and_probs, play_data)]
        
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

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value,
                                         c_puct=self.c_puct,
                                         n_playout=50, name="Current")
        old_mcts_player = MCTSPlayer(self.duplicate_policy_value_net.policy_value,
                                         c_puct=self.c_puct,
                                         n_playout=400, name="Old")
        win_cnt = {None: 0, current_mcts_player: 0, old_mcts_player: 0}
        for i in range(n_games):
            if i % 2 == 0:
                p1, p2 = current_mcts_player, old_mcts_player
            else:
                p1, p2 = old_mcts_player, current_mcts_player
            winner = self.game.start_play(p1, p2, is_shown=True)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[current_mcts_player] + 0.5*win_cnt[None]) / n_games
        print("win: {}, lose: {}, tie:{}".format(win_cnt[current_mcts_player], win_cnt[old_mcts_player], win_cnt[None]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            self.backup_policy()
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                for i in range(self.num_updates):
                    self.policy_update()

                if (i+1) % self.check_freq == 0:
                    win_ratio = self.policy_evaluate(self.eval_n_games)
                    if win_ratio >= self.win_ratio_cutoff:
                        print("New best policy!!!!!!!!")
                        self.policy_value_net.save_model(best_model_path)
                        self.backup_policy()

        except KeyboardInterrupt:
            print('\n\rquit')


import os

best_model_path = "models/best_policy.model"

if __name__ == '__main__':
    if os.path.isfile(best_model_path+".meta"):
        best_model_file = best_model_path
    else:
        best_model_file = None
    training_pipeline = TrainPipeline(init_model=best_model_file)
    training_pipeline.run()
