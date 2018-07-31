#ifndef MOVE_HPP
#define MOVE_HPP

#include "mcts.h"
#include "Piece.h"

namespace elder_chess {

class Move final /* : public Action<Board> */ {
public:
	enum class Type : unsigned int {
		FLIP = 0,
		UP = 1,
		DOWN = 2,
		LEFT = 3,
		RIGHT = 4,
		NONE = 5,
		ENV_RAND = 6
	};

	union {
		struct {
			unsigned int x;
			unsigned int y;
		};
		Piece potential_piece;
	};
	
	Type type = Type::FLIP;

	Move() :
		x(0), y(0), type(Type::FLIP) 
	{ }

	Move(Type type, unsigned int x, unsigned int y):
		type(type),
		x(x), y(y)
	{ }

	Move(Piece p): 
		type(Type::ENV_RAND),
		potential_piece(p) 
	{ }

	Move(const std::string& m_str) {
		std::istringstream strm(m_str);
		*this = Move::parse(strm);
	}

	void print(std::ostream &os);

	static Move parse(std::istream &in); 

	inline size_t hash() const {
		return ((((size_t)x << 16) ^ (size_t)y) << 32) ^ ((size_t)type);
	}

	inline size_t hash() {
		return hash();
	}

	inline bool operator==(const Move& a) const {
		return x == a.x && y == a.y && type == a.type;
	}

	inline bool operator!=(const Move& a) const {
		return !(*this == a);
	}

	inline bool operator<(const Move& b) const {
		return hash() < b.hash();
	}
};

inline std::ostream& operator<<(std::ostream& os, Move move) {
	move.print(os);
	return os;
}

}

namespace std {
	template <> struct hash<elder_chess::Move> {
		size_t operator()(const elder_chess::Move& x) const {
			return x.hash();
		}
	};
}

#endif