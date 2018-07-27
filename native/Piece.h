#ifndef PIECE_H
#define PIECE_H

#include <cassert>

namespace elder_chess {

typedef int Side;

struct Sides {
	const static constexpr Side PLAYER_0 = 0;
	const static constexpr Side PLAYER_1 = 1;
	const static constexpr Side NONE = -1;
	const static constexpr Side DRAW = 2;
	inline const static Side opponent(Side s) {
		assert(s == PLAYER_0 || s == PLAYER_1);
		return 1 - s;
	}
};

enum class PieceType : unsigned int {
	EMPTY =  0x0, // 0b00,
	HIDDEN = 0x2, // 0b10,
	PLAYER_0 = 0x3, // 0b11,
	PLAYER_1 = 0x1 // 0b01
};

struct Piece final {

	PieceType type; 
	unsigned int value;

	Piece(bool empty=true):
		type(empty ? PieceType::EMPTY : PieceType::HIDDEN) 
	{ }

	Piece(PieceType type, unsigned int value):
		type(type),
		value(value)
	{ }

	Piece(Side side, unsigned int value):
		type(side == Sides::PLAYER_0 ? PieceType::PLAYER_0 : PieceType::PLAYER_1),
		value(value)
	{ }

	Piece& operator=(const Piece& other) = default;

	static inline Piece empty() { 
		return Piece(true);
	}

	static inline Piece hidden() { 
		return Piece(false); 
	}

	inline Side getSide() const {
		if(type == PieceType::PLAYER_0) {
			return Sides::PLAYER_0;
		} else if( type == PieceType::PLAYER_1) {
			return Sides::PLAYER_1;
		} else {
			return Sides::NONE;
		}
	}

	inline bool isHidden() const {
		return type == PieceType::HIDDEN;
	}

	inline bool isEmpty() const {
		return type == PieceType::EMPTY;
	}

	inline bool canEat(Piece other) const {
		if(isEmpty() || isHidden() || other.isHidden()) {
			return false;
		} else if (other.isEmpty()) {
			return true;
		} else if (getSide() == other.getSide()) {
			return false;
		} else {
			if (value == 0 && other.value == 3) {
				return true;
			} else if(value == 3 && other.value == 0) {
				return false;
			} else {
				return value >= other.value;
			}
		}
	}

	size_t hash() const {
		return (((size_t)type) << 32) ^ ((size_t)value);
	}

	inline bool operator==(const Piece& a) const {
		return type == a.type && value == a.value;
	}

	inline bool operator!=(const Piece& a) const {
		return !(*this == a);
	}


	friend std::ostream& operator<<(std::ostream& os, const Piece& p){
		switch(p.type) {
			case PieceType::EMPTY: { os << "  "; break; }
			case PieceType::HIDDEN: { os << " X"; break; }
			case PieceType::PLAYER_0: { os << "W" << p.value + 1; break; }
			case PieceType::PLAYER_1: { os << "B" << p.value + 1; break; }
		}
	    return os;
	};
};

}

namespace std {
	template <> struct hash<elder_chess::Piece> {
		size_t operator()(const elder_chess::Piece& x) const {
			return x.hash();
		}
	};
}

#endif
