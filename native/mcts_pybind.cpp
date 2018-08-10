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

std::mt19937 rng(time(0));

typedef Board<false> Board_;

static void fill_compact_state(const Board_& board, double (&board_state)[9][4][4], double (&hiddens_state)[2][4], double &remaining_steps_state) {
    memset(&board_state, 0, sizeof(board_state));
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            Piece p = board.at(i, j);
            if (p.isHidden()) {
                board_state[8][i][j] = 1.;
            } else if (!p.isEmpty()) {
                int idx = p.getSide() == board.get_current_player() ? 0 : 1;
                board_state[idx * 4 + p.value][i][j] = 1.;
            }
        }
    }

    auto&& counts = board.get_hidden_counts();
    for(int i = 0; i < 4; i++) {
        hiddens_state[board.get_current_player() == 0 ? 0 : 1][i] = counts.first[i];
        hiddens_state[board.get_current_player() == 1 ? 0 : 1][i] = counts.second[i];
    }
    
    remaining_steps_state = board.remaining_steps();
}

typedef std::tuple<py::array_t<double>, py::array_t<double>, double> CompactState;

static CompactState get_compact_state(const Board_& board) {
    py::array_t<double> ret({ 9, 4, 4 });
    py::array_t<double> hiddens({ 2, 4 });
    double remaining_steps;
    
    double (&b1)[9][4][4] = (double (&)[9][4][4])*ret.mutable_unchecked<3>().mutable_data(0, 0, 0);
    double (&b2)[2][4] = (double (&)[2][4])*hiddens.mutable_unchecked<2>().mutable_data(0, 0);

    fill_compact_state(board, b1, b2, remaining_steps);

    return std::make_tuple(ret, hiddens, remaining_steps);
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

    typedef std::function<std::pair<py::array_t<double>, py::array_t<double>>(std::tuple<py::object, py::object, py::object>)> PolicyNetworkF;

    py::class_<MCTS<Board_>>(m, "MCTS")
        // .def(py::init<const MCTS<Board_>::PolicyFunction&, double, unsigned int>())
        .def(py::init([](const PolicyNetworkF& policy_f, double c_puct, int n_playout, int thread_pool_size, int eval_batch_size) {
            return new MCTS<Board_>(
                [policy_f, eval_batch_size]
                (const std::vector<Board_>& boards, std::vector<MCTS<Board_>::EvalResult>& results, int batch_size, void* buffer) {
                    double* _board_states = (double*)buffer;
                    double* _hiddens_states = _board_states + eval_batch_size * 9 * 4 * 4;
                    double* _remaining_steps_states = _hiddens_states + eval_batch_size * 2 * 4;
                    for(int i = 0; i < batch_size; i++) {
                        fill_compact_state(boards[i], (double(&)[9][4][4])_board_states[i * 9 * 4 * 4], (double(&)[2][4])_hiddens_states[i * 2 * 4], _remaining_steps_states[i]);
                    }

                    {
                        py::gil_scoped_acquire acquire;

                        py::object board_states = py::array_t<double, py::array::c_style>(
                            {batch_size, 9, 4, 4}, 
                            _board_states
                        );
                        py::object hiddens_states = py::array_t<double, py::array::c_style>(
                            {batch_size, 2, 4},
                            _hiddens_states
                        );
                        py::object remaining_steps_states = py::array_t<double, py::array::c_style>({batch_size, 1}, _remaining_steps_states);
                        auto policy_input = std::make_tuple(board_states, hiddens_states, remaining_steps_states);
                        
                        auto&& move_probs_and_value = policy_f(policy_input);

                        auto move_probs_buf = move_probs_and_value.first.unchecked<2>();
                        auto values_buf = move_probs_and_value.second.unchecked<2>();

                        for(int i = 0; i < batch_size; i++) {
                            assert(!boards[i].is_env_move());
                            std::vector<Move>&& available_moves = boards[i].get_moves();
                            MCTS<Board_>::EvalResult& result = results[i];
                            result.first.resize(available_moves.size());
                            for(int j = 0; j < available_moves.size(); j++) {
                                Move m = available_moves[j];
                                result.first[j] = std::make_pair(m, move_probs_buf(i, m.y + m.x * 4 + ((int)m.type) * 4 * 4));
                                result.second = values_buf(i, 0);
                            }
                        }
                    }
                },
                (9 * 4 * 4) + (2 * 4) + (1),
                c_puct,
                n_playout,
                thread_pool_size,
                eval_batch_size
            );
        }))
        .def("get_move_counts", &MCTS<Board_>::get_move_counts, py::call_guard<py::gil_scoped_release>())
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
    )
    .def("random_seed", [](double seed) {
        rng.seed(seed);
    });
}