#ifndef MCTS_H
#define MCTS_H

#include <stdexcept>
#include <sstream>
#include <random>
#include <iostream>
#include <atomic>
#include <cstdint>

#include "threading.hpp" 

namespace mcts {

template<typename> class MCTS;

template<typename State>
class TreeNode
{
public:
	typedef typename State::Move Move;
	friend class MCTS<State>;

	TreeNode(TreeNode<State>* parent, double _prior) : 
		_parent(parent),
		_prior(_prior)
	{ }

	~TreeNode();

	void expand(const std::vector<std::pair<Move, double>>& priors);

	template<typename RandomEngine>
	std::pair<Move, TreeNode<State>*> select(double c_puct, RandomEngine*) const;

	template<typename RandomEngine>
	std::pair<Move, TreeNode<State>*> env_select(RandomEngine*) const;
	
	double get_U_value(double c_puct) const;
	bool is_leaf() const;
	bool is_root() const;
	void update(double leaf_value);

    void lock();
    void unlock();
    void add_virtual_loss();
    void remove_virtual_loss();

    /* Debug Utils */
	void debug_check_n_visit();

protected:
	TreeNode<State>* const _parent;
	std::vector<std::pair<Move, TreeNode<State>*>> _children;
	std::atomic<std::uint16_t> _n_visit = { 0 };
	std::atomic<std::int16_t> _virtual_loss = { 0 };
	std::atomic<double> _W = { 0 };
	double _prior;

	threading::SpinLock _lock;
};

#include "TreeNode.ipp"

template<typename State>
class MCTS
{
public:
	typedef typename State::Move Move;
	typedef std::pair<std::vector<std::pair<typename State::Move, double>>, double> EvalResult;

	typedef std::function<void(const std::vector<State>& boards, std::vector<EvalResult>&, int, void*)> PolicyFunction;

	MCTS(const PolicyFunction& policy_fn, std::size_t compact_state_size, double c_puct, std::size_t n_playout, std::size_t thread_pool_size, std::size_t eval_batch_size);

	~MCTS();

	std::pair<std::vector<typename State::Move>, std::vector<std::uint16_t>> get_move_counts(State& state);

    void update_with_move_index(State curState, std::uint16_t move_index);
    void update_with_move(const State& nextState, Move move);

    void reset();

private:

	template<typename RandomEngine>
	TreeNode<State>* _playout_single_path(State& state, std::vector<int>& players, double& leaf_value, bool& game_ended, RandomEngine* rng);

	template<typename RandomEngine>
	void _playout_batch(const State& state, std::size_t n_playout, RandomEngine* rng);

	void _backprop_single_path(TreeNode<State>* node, double leaf_value, const std::vector<int>& players);

	void _remove_virtual_losses(TreeNode<State>* node);

	int _eval_and_backprop_batch(const std::vector<TreeNode<State>*>& nodes, 
		const std::vector<State>& states, 
		const std::vector<std::vector<int>>& players, 
		const std::vector<double>& batch_ended_results,
    	const std::vector<double>& compact_state_buffer,
		std::vector<MCTS<State>::EvalResult>& eval_results, 
		int batch_size
	);

	TreeNode<State>* _root;
	TreeNode<State>* _current_root;
	const PolicyFunction _policy_fn;
	std::size_t _compact_state_size;

	double _c_puct;
	std::size_t _n_playout;

	std::size_t _eval_batch_size; 
	std::size_t _thread_pool_size;

	threading::ThreadPool _pool;
};

#include "mcts.ipp"

}
#endif