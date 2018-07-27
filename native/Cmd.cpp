#include "Board.h"
// #include "MCTSPlayer.h"
#include <vector>
#include <random>

int main(int argc, char const *argv[])
{
	std::mt19937_64 random_engine(time(NULL));

	Board board;
	std::vector<Move> firstMoves;
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			firstMoves.push_back(Move(Move::Type::FLIP, i, j));
		}
	}
	// for(auto move : firstMoves) {
	// 	board.move(move);
	// 	if(move.type == Move::Type::FLIP) {
	// 		Move env_move = board.chooseRandomFromHidden(&random_engine);
	// 		board.move(env_move);
	// 	}
	// }
	firstMoves.clear();
	bool human_player = true;

	std::cout << board << std::endl; 

	MCTS::ComputeOptions player1_options, player2_options;
	player1_options.max_iterations = -1;
	player1_options.max_time = 10;
	player1_options.verbose = true;
	player2_options.max_iterations = -1;
	player2_options.max_time = 10;
	player2_options.verbose = true;

	int turn = 0;
	while(true) {
		Side winner = board.getWinner();
		switch(winner) {
			case Sides::PLAYER_0: {
				std::cout << "Player W wins!" << std::endl;
				return 0;
			}
			case Sides::PLAYER_1: {
				std::cout << "Player B wins!" << std::endl;
				return 0;
			}
			case Sides::DRAW: {
				std::cout << "Game draws!" << std::endl;
				return 0;
			}
		}
		Move m;
		if(turn % 2 == 1) {
			m = MCTS::compute_move(board, player1_options);
			std::cout << "AI: " << m << std::endl;
		} else {
			if(human_player) {
				if(firstMoves.size() > 0) {
					m = firstMoves.back();
					firstMoves.pop_back();
				} else {
					try {
						m = Move::parse(std::cin);
					} catch (std::invalid_argument& iae) {
						continue;
					}
				}
			} else {
				m = MCTS::compute_move(board, player2_options);
			}
		}
		if(!board.do_move_with_env_safe(m, &random_engine)) {
			std::cout << "Invalid move" << std::endl;
			continue;
		}
		std::cout << board << std::endl;
		turn ++;
	}
}