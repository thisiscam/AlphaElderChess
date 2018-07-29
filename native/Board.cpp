#include "Board.h"
#include <stdlib.h>
#include <stdexcept>

namespace elder_chess {

Board::Board(int maxSteps) :
	maxSteps(maxSteps),
	hiddenPiecesCount(Board::SIDE * Board::SIDE) 
{
	for(int i = 0; i < Board::SIDE; i++) {
		for(int j = 0; j < Board::SIDE; j++) {
			board[i][j] = Piece::hidden();
		}
	}
	for(int i = 0; i < 4; i++) {
		hiddenPieces.push_back(Piece(Sides::PLAYER_0, i));
		hiddenPiecesCounts.push_back(2);
		hiddenPieces.push_back(Piece(Sides::PLAYER_1, i));
		hiddenPiecesCounts.push_back(2);

		onBoardPieces[Sides::PLAYER_0][i] = 2;
		onBoardPieces[Sides::PLAYER_1][i] = 2;
	}
}

void
Board::print(std::ostream &os) const {
	os << "turn " << steps << "/" << maxSteps << std::endl;
	os << (get_current_player() == Sides::PLAYER_0 ? "W" : "B") << " | 0| 1| 2| 3|" << std::endl;
	for(int i = 0; i < 4; i++) {
		os << i << " |";
		for(int j = 0; j < 4; j++) {
			os << at(i, j) << "|";
		}
		os << std::endl;
	}
	os << "W[1~4]:"; 
	for(int i = 0; i < 4; i++) {
		os << onBoardPieces[Sides::PLAYER_0][i] << "|";
	}
	os << std::endl;
	os << "B[1~4]:";
	for(int i = 0; i < 4; i++) {
		os << onBoardPieces[Sides::PLAYER_1][i] << "|";
	}
	os << std::endl;
}

const Move Board::no_move = Move(Move::Type::NONE, -1, -1);

}