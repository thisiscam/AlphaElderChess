#include "Board.h"
#include "mcts.h"

#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

using namespace mcts;
using namespace elder_chess;

namespace py = pybind11;

std::default_random_engine mcts::rng(time(0));

typedef std::tuple<py::array_t<double>, py::array_t<double>, double> CompactState;

typedef Board<false> Board_;

static CompactState get_compact_state(const Board_& board) {
	py::array_t<double> ret({ 9, 4, 4 });
	auto buf = ret.mutable_unchecked<3>();
	memset(buf.mutable_data(0, 0, 0), 0, buf.nbytes());
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			Piece p = board.at(i, j);
			if (p.isHidden()) {
				buf(8, i, j) = 1.;
			} else if (!p.isEmpty()) {
				int idx = p.getSide() == board.get_current_player() ? 0 : 1;
				buf(idx * 4 + p.value, i, j) = 1.;
			}
		}
	}
	py::array_t<double> hiddens({2, 4});
	auto buf2 = hiddens.mutable_unchecked<2>();
	auto&& counts = board.get_hidden_counts();
	for(int i = 0; i < 4; i++) {
		buf2(board.get_current_player() == 0 ? 0 : 1, i) = counts.first[i];
		buf2(board.get_current_player() == 1 ? 0 : 1, i) = counts.second[i];
	}
	return std::make_tuple(ret, hiddens, board.remaining_steps());
}

PYBIND11_MODULE(elder_chess_native, m) {
	py::class_<Board_>(m, "Board")
		.def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const Board_&>())
		.def("do_move", &Board_::do_move)
		.def("get_moves", &Board_::get_moves)
		.def("get_winner", [](const Board_ &board) {
			return (int)board.get_winner();
		})
		.def("get_current_player", &Board_::get_current_player)
		.def("is_env_move", &Board_::is_env_move)
		.def("game_ended", &Board_::game_ended)
		.def("env_do_move", [](Board_& board) { return board.env_do_move(&rng); })
		.def("do_move_with_env", [](Board_& board, Move m) { return board.do_move_with_env(m, &rng); })
		.def("do_move_with_env_safe", [](Board_& board, Move m) { return board.do_move_with_env_safe(m, &rng); })
		.def("do_move_safe", [](Board_& board, Move m) { return board.do_move_safe(m, &rng); })
		.def("get_compact_state", &get_compact_state)
		.def("__str__", [](const Board_ &board) {
			std::ostringstream stream;
		    stream << board;
		    return stream.str();
		})
		.def("remaining_steps", &Board_::remaining_steps)
		.def("get_moves_one_hot", [](const Board_& board) {
			std::vector<Move> moves = board.get_moves();
			py::array_t<unsigned int> ret({5, 4, 4});
			auto buf = ret.mutable_unchecked<3>();
			memset(buf.mutable_data(0, 0, 0), 0, buf.nbytes());
			for(Move m : moves) {
				buf(m.type, m.x, m.y) = 1;
			}
			return ret;
		})
	;

	py::class_<Move>(m, "Move")
		.def(py::init<const std::string&>())
		.def(py::init([](int type, unsigned int x, unsigned int y) {
			return new Move((Move::Type)type, x, y);
		}))
		.def(py::self == py::self)
		.def("by_env", [](const Move& move) { return move.type != Move::Type::ENV_RAND; })
		.def("__str__", [](const Move &move) {
			std::ostringstream stream;
		    stream << move;
		    return stream.str();
		})
	;

	typedef std::function<std::pair<py::array_t<double>, double>(const CompactState&)> PolicyNetworkF;

    py::class_<MCTS<Board_>>(m, "MCTS")
        // .def(py::init<const MCTS<Board_>::PolicyFunction&, double, unsigned int>())
        .def(py::init([](const PolicyNetworkF& policy_f, double c_puct, unsigned int n_playout) {
        	return new MCTS<Board_>(
        		[policy_f](const Board_& b) {
        			auto&& move_probs_and_value = policy_f(get_compact_state(b));
        			py::array_t<double> move_probs = move_probs_and_value.first;
        			double value = move_probs_and_value.second;
        			auto move_probs_buf = move_probs.unchecked<1>();
        			std::vector<Move> available_moves = b.get_moves();
					std::vector<std::pair<Move, double>> available_moves_probs(available_moves.size());
					for(int i = 0; i < available_moves.size(); i++) {
						Move m = available_moves[i];
						available_moves_probs[i] = 
							std::make_pair(m, move_probs_buf(m.y + m.x * 4 + ((int)m.type) * 4 * 4));
					}
					return std::make_pair(available_moves_probs, value);
        		},
        		c_puct,
        		n_playout
        	);
        }))
        .def("get_move_probs", &MCTS<Board_>::get_move_probs)
        .def("update_with_move", &MCTS<Board_>::update_with_move)
        .def("update_with_move_index", &MCTS<Board_>::update_with_move_index)
        .def("reset", &MCTS<Board_>::reset)
    ;

    m.def("move_probs_to_one_hot", 
    	[](const std::vector<Board_::Move>& moves, const std::vector<double>& probs) {
			py::array_t<double> ret({5, 4, 4});
			auto buf = ret.mutable_unchecked<3>();
			memset(buf.mutable_data(0, 0, 0), 0, buf.nbytes());
			for(int i = 0; i < moves.size(); i++) {
				Move m = moves[i];
				buf(m.type, m.x, m.y) = probs[i];
			}
			return ret;
    	}
    );

}