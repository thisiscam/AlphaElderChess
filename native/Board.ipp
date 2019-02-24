template<bool ds>
bool Board<ds>::_canEat(const Piece& from, const Piece& to) const {
	if(from.value == 0 && to.value == 3) {
		return true;
	} if(from.value == 3 && to.value == 0) {
		return false;
	} else if (from.value >= to.value) {
		return true;
	} else {
		return false;
	}
}

template<bool ds>
bool Board<ds>::_piecesDominating(Side player) const {
	if(onBoardPieces[player][0] > 0) {
		if(onBoardPieces[1 - player][0] == 0 && onBoardPieces[1 - player][1] == 0 && onBoardPieces[1 - player][2] == 0) {
			return true;
		}
	}
	if(onBoardPieces[player][1] > 0) {
		if(onBoardPieces[1 - player][1] == 0 && onBoardPieces[1 - player][2] == 0 && onBoardPieces[1 - player][3] == 0) {
			return true;
		}
	}
	if(onBoardPieces[player][2] > 0) {
		if(onBoardPieces[1 - player][3] == 0 && onBoardPieces[1 - player][2] == 0) {
			return true;
		}
	}
	if(onBoardPieces[player][3] > 0) {
		if(onBoardPieces[1 - player][0] == 0 && onBoardPieces[1 - player][3] == 0) {
			return true;
		}
	}
	return false;
}

template<bool ds>
Side Board<ds>::get_winner() const {
	std::vector<Move> p0Moves = _scanAvailableMoves(Sides::PLAYER_0);
	if(p0Moves.size() == 0) {
		return Sides::PLAYER_1;
	}
	std::vector<Move> p1Moves = _scanAvailableMoves(Sides::PLAYER_1);
	if(p1Moves.size() == 0) {
		return Sides::PLAYER_0;
	}
	if(_piecesDominating(Sides::PLAYER_0)) {
		return Sides::PLAYER_0;
	}
	if(_piecesDominating(Sides::PLAYER_1)) {
		return Sides::PLAYER_1;
	}
	if(get_remaining_steps() == 0) {
		if(p1Moves.size() > p0Moves.size()) {
			return Sides::PLAYER_1;
		}
		if(p1Moves.size() < p0Moves.size()) {
			return Sides::PLAYER_0;
		}
		return Sides::DRAW;
	}
	// TODO
	return Sides::NONE;
}

template<bool ds>
bool Board<ds>::_currentIsEnvironment() const {
	return get_current_player() < 0;
}


/*
	Assumes that board[i][j].side == currentPlayer
*/
template<bool ds>
bool Board<ds>::_checkMoveable(int i, int j, int di, int dj) const {
	int ti = i + di;
	int tj = j + dj;
	if(ti >= 0 && ti < SIDE && tj >= 0 && tj < SIDE) {
		Piece from = board[i][j];
		Piece to = board[ti][tj];
		if(to.isEmpty()) {
			return true;
		} else if(to.isHidden()) {
			return false;
		} else if(from.getSide() == to.getSide()) {
			return false;
		} else {
			return _canEat(from, to);
		}
	} else {
		return false;
	}
}

template<bool ds>
std::vector<Move> Board<ds>::_scanAvailableMoves(Side side) const { 
	assert(side >= 0);
	std::vector<Move> moves;
	for(int i=0; i < SIDE; i++) {
		for(int j=0; j < SIDE; j++) {
			if(board[i][j].isHidden()) {
				moves.push_back(Move(Move::Type::FLIP, i, j));
			} else if (!board[i][j].isEmpty() && board[i][j].getSide() == side) {
				if(_checkMoveable(i, j, -1, 0)) {
					moves.push_back(Move(Move::Type::UP, i, j));
				}
				if(_checkMoveable(i, j, +1, 0)) {
					moves.push_back(Move(Move::Type::DOWN, i, j));
				}
				if(_checkMoveable(i, j, 0, -1)) {
					moves.push_back(Move(Move::Type::LEFT, i, j));
				}
				if(_checkMoveable(i, j, 0, +1)) {
					moves.push_back(Move(Move::Type::RIGHT, i, j));
				}
			}
		}
	}
	return moves;
}

template<bool ds>
void Board<ds>::_removeHidden(Piece p) {
	int foundIdx = -1;
	for(int i = 0; i < hiddenPieces.size(); i++) {
		if(hiddenPieces[i] == p) {
			if(hiddenPiecesCounts[i] > 1) {
				hiddenPiecesCounts[i] -= 1;
			} else {
				foundIdx = i;
			}
			hiddenPiecesCount -= 1;
			break;
		}
	}
	if(foundIdx >= 0) {
		hiddenPieces[foundIdx] = hiddenPieces[hiddenPieces.size() - 1];
		hiddenPiecesCounts[foundIdx] = hiddenPiecesCounts[hiddenPiecesCounts.size() - 1];
		hiddenPieces.pop_back();
		hiddenPiecesCounts.pop_back();
	}
}

template<bool ds>
void Board<ds>::_sync_on_board(int x, int y) {
	Piece p = board[x][y];
	if(!p.isEmpty()) {
		onBoardPieces[p.getSide()][p.value] --;
		if(ds) {
			remaining_steps = maxSteps;
		}
	}
}

template<bool ds>
void Board<ds>::do_move(Move m) {
	switch(m.type) {
		case Move::Type::ENV_RAND: {
			int x = about_to_flip.x;
			int y = about_to_flip.y;
			Piece p = m.potential_piece;
			_removeHidden(p);
			about_to_flip = Move(Move::Type::NONE, 0, 0);
			player_to_move = 1 - (- (player_to_move + 1));
			board[x][y] = p;
			if(ds) {
				remaining_steps = maxSteps - 1;
			}
			break;
		}
		case Move::Type::FLIP: {
			about_to_flip = m;
			player_to_move = - player_to_move - 1;
			if(ds) {
				remaining_steps = maxSteps;
			}
			steps++;
			break;
		}
		case Move::Type::UP: {
			_sync_on_board(m.x - 1, m.y);
			board[m.x - 1][m.y] = board[m.x][m.y];
			board[m.x][m.y] = Piece::empty();
			player_to_move = 1 - player_to_move;
			steps++;
			remaining_steps--;
			break;
		}
		case Move::Type::DOWN: {
			_sync_on_board(m.x + 1, m.y);
			board[m.x + 1][m.y] = board[m.x][m.y];
			board[m.x][m.y] = Piece::empty();
			player_to_move = 1 - player_to_move;
			steps++;
			remaining_steps--;
			break;
		}
		case Move::Type::LEFT: {
			_sync_on_board(m.x, m.y - 1);
			board[m.x][m.y - 1] = board[m.x][m.y];
			board[m.x][m.y] = Piece::empty();
			player_to_move = 1 - player_to_move;
			steps++;
			remaining_steps--;
			break;
		}
		case Move::Type::RIGHT: {
			_sync_on_board(m.x, m.y + 1);
			board[m.x][m.y + 1] = board[m.x][m.y];
			board[m.x][m.y] = Piece::empty();
			player_to_move = 1 - player_to_move;
			steps++;
			remaining_steps--;
			break;
		}
		default: {
			std::cout << m << std::endl;
			throw std::invalid_argument("invalid m type");
		}
	}
}

template<bool ds>
std::vector<Move> Board<ds>::get_moves() const {
	if(_currentIsEnvironment()) {
		std::vector<Move> moves;
		std::transform(
			hiddenPieces.begin(), 
			hiddenPieces.end(), 
			std::back_inserter(moves), 
			[](Piece p) { return Move(p); }
		);
		return moves;
	} else {
		return _scanAvailableMoves(get_current_player());
	}
}

template<bool ds>
std::vector<std::pair<Move, double>> Board<ds>::get_env_move_weights() const {
	assert(_currentIsEnvironment());
	std::vector<std::pair<Move, double>> weights(hiddenPieces.size());
	for(int i = 0; i < hiddenPieces.size(); i++) {
		weights[i] = std::make_pair(Move(hiddenPieces[i]), (double)hiddenPiecesCounts[i]);
	}
	return weights;
}

template<bool ds>
bool Board<ds>::is_env_move() const {
	return player_to_move < 0;
}

template<bool ds>
bool Board<ds>::game_ended() const {
	return get_winner() != Sides::NONE;
}

template<bool ds>
template<typename RandomEngine>
Move Board<ds>::env_do_move(RandomEngine* engine) {
	assert(_currentIsEnvironment());
	std::uniform_int_distribution<int> rnds(0, hiddenPiecesCount - 1);
    int rnd = rnds(*engine);
    Piece p;
    for(int i=0; i < hiddenPieces.size(); i++) {
        if(rnd < hiddenPiecesCounts[i]) {
            p = hiddenPieces[i];
            break;
        }
        rnd -= hiddenPiecesCounts[i];
    }
    Move m = Move(p);
    do_move(m);
    return m;
}

template<bool ds>
template<typename RandomEngine>
void Board<ds>::do_move_with_env(Move m, RandomEngine *engine) {
	do_move(m);
	if(m.type == Move::Type::FLIP) {
		env_do_move(engine);
	}
}

template<bool ds>
template<typename RandomEngine>
bool Board<ds>::do_move_with_env_safe(Move m, RandomEngine* engine) {
	assert(!_currentIsEnvironment());
	std::vector<Move> moves = _scanAvailableMoves(get_current_player());
	if(std::find(moves.begin(), moves.end(), m) == moves.end()) {
		return false;
	} else {
		do_move_with_env(m, engine);
		return true;
	}
}

template<bool ds>
template<typename RandomEngine>
bool Board<ds>::do_move_safe(Move m, RandomEngine* engine) {
	assert(!_currentIsEnvironment());
	std::vector<Move> moves = _scanAvailableMoves(get_current_player());
	if(std::find(moves.begin(), moves.end(), m) == moves.end()) {
		return false;
	} else {
		do_move(m);
		return true;
	}
}

template<bool ds>
template<typename RandomEngine>
Move Board<ds>::do_random_move(RandomEngine *engine) {
	if(_currentIsEnvironment()) {
		return env_do_move(engine);
	} else {
		std::vector<Move> moves = _scanAvailableMoves(get_current_player());
		std::uniform_int_distribution<int> rnds(0, moves.size() - 1);
        Move m = moves[rnds(*engine)];
        do_move(m);
        return m;
	}
}

template<bool ds>
Board<ds>::Board(int maxSteps) :
	maxSteps(maxSteps),
	remaining_steps(maxSteps),
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

template<bool ds>
void Board<ds>::print(std::ostream &os) const {
	os << "turn " << steps << "  Rem: " << get_remaining_steps() <<std::endl;
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

template<bool ds>
const Move Board<ds>::no_move = Move(Move::Type::NONE, -1, -1);

template<bool ds>
inline std::ostream& operator<<(std::ostream& os, const Board<ds>& board) {
	board.print(os);
	return os;
}
