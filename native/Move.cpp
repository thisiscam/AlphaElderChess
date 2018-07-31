#include <stdexcept>

#include "Move.h"
#include "Board.h"

namespace elder_chess {

void
Move::print(std::ostream &os) {
	if(type == Move::Type::FLIP) {
		os << "Flip (" << x << "," << y << ")";
	} else if(type == Move::Type::NONE) {
		os << "None";;
	} else if(type == Move::Type::ENV_RAND) {
		os << "F<" << potential_piece << ">";
	}else {
		os << "Move (" << x << "," << y << ") ";
		switch(type) {
			case Move::Type::UP: {
				os << "up";
				break;
			}
			case Move::Type::DOWN: {
				os << "down";
				break;
			}
			case Move::Type::LEFT: {
				os << "left";
				break;
			}
			case Move::Type::RIGHT: {
				os << "right";
				break;
			}
			default: {
				break;
			}
		}
	}
}

Move
Move::parse(std::istream& in) {
	std::string typeStr;
	unsigned int x, y;
	in >> typeStr >> x >> y;
	if(typeStr[0] == 'f' || typeStr[0] == 'F') {
		return Move(Move::Type::FLIP, x, y);
	} else if (typeStr[0] == 'M' || typeStr[0] == 'm') {
		std::string dirStr;
		in >> dirStr;
		if(dirStr[0] == 'u' || dirStr[0] == 'U') {
			return Move(Move::Type::UP, x, y);
		} else if(dirStr[0] == 'd' || dirStr[0] == 'D') {
			return Move(Move::Type::DOWN, x, y);
		} else if (dirStr[0] == 'l' || dirStr[0] == 'L') {
			return Move(Move::Type::LEFT, x, y);
		} else if (dirStr[0] == 'r' || dirStr[0] == 'R') {
			return Move(Move::Type::RIGHT, x, y);
		} else {
			throw std::invalid_argument("move direction");
		}
	} else {
		throw std::invalid_argument("move type");
	}
}

}