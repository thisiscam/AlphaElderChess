#ifndef BOARD_H
#define BOARD_H

#include <iostream>
#include <vector>
#include <unordered_set>

#include "mcts.h"

#include "hashed_vector.hpp"
#include "Piece.h"
#include "Move.h"

namespace elder_chess {

template<bool dynamic_steps>
class Board final /* : public State */ {

public:
	
	using Move = Move;

	static const Move no_move;

	static const int DEFAULT_MAX_STEPS = dynamic_steps ? 8 : 40 ;
	static const int SIDE = 4;

	Board():Board(DEFAULT_MAX_STEPS) { }

	Board(int maxSteps);

	Board(const Board& other) = default;

	Board& operator=(const Board& other) = default;
	
	void print(std::ostream &strm) const;

	inline Side get_current_player() const {
		return player_to_move;
	}

	inline Piece& at(int i, int j) {
		return board[i][j];
	}

	inline const Piece& at(int i, int j) const {
		return board[i][j];
	}

	inline const int get_remaining_steps() const {
		if(dynamic_steps) {
			return remaining_steps;
		} else {
			return maxSteps - steps;
		}
	}

	inline const int get_total_steps() const {
		return steps;
	}

	inline void do_move(Move m);

	inline std::vector<Move> get_moves() const;

	inline std::vector<std::pair<Move, double>> get_env_move_weights() const;

	inline bool is_env_move() const;

	inline bool game_ended() const;

	inline Side get_winner() const;

	template<typename RandomEngine>
	Move env_do_move(RandomEngine* engine);

	template<typename RandomEngine>
	void do_move_with_env(Move m, RandomEngine *engine);

	template<typename RandomEngine>
	bool do_move_with_env_safe(Move m, RandomEngine* engine);

	template<typename RandomEngine>
	bool do_move_safe(Move m, RandomEngine* engine);

	template<typename RandomEngine> 
	Move do_random_move(RandomEngine *engine);

	inline std::pair<int[4], int[4]> get_hidden_counts() const {
		std::pair<int[4], int[4]> ret;
		for(int i = 0; i < hiddenPieces.size(); i++) {
			if(hiddenPieces[i].getSide() == 0) {
				ret.first[hiddenPieces[i].value] = hiddenPiecesCounts[i];
			} else {
				ret.second[hiddenPieces[i].value] = hiddenPiecesCounts[i];
			}
		}
		return ret;
	}

private:

	inline bool _canEat(const Piece& from, const Piece& to) const;

	inline bool _piecesDominating(Side player) const;

	inline bool _currentIsEnvironment() const;

	/*
		Assumes that board[i][j].side == currentPlayer
	*/
	inline bool _checkMoveable(int i, int j, int di, int dj) const;

	inline std::vector<Move> _scanAvailableMoves(Side side) const;

	inline void _removeHidden(Piece p);

	inline void _sync_on_board(int x, int y);

	int player_to_move = 0;
	Move about_to_flip = Move(Move::Type::NONE, 0, 0);

	int steps = 0;
	int remaining_steps;

	int maxSteps;
	Piece board[4][4];

	std::vector<Piece> hiddenPieces;
	std::vector<int> hiddenPiecesCounts;
	int hiddenPiecesCount; // sum of the above vector

	int onBoardPieces[2][4];
};

#include "Board.ipp"

}

#endif