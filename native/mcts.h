#ifndef MCTS_H
#define MCTS_H

#include <stdexcept>
#include <sstream>
#include <random>
#include <iostream>

#include "threading.hpp"

namespace mcts {

extern std::mt19937 rng;

template<typename> class MCTS;
template<typename> class BatchMCTS;

template<typename State>
class TreeNode
{
public:
	typedef typename State::Move Move;
	friend class MCTS<State>;
	friend class BatchMCTS<State>;

	TreeNode(TreeNode<State>* parent, double _prior) : 
		_parent(parent),
		_prior(_prior)
	{ }

	~TreeNode();

	void expand(std::vector<std::pair<Move, double>> priors);

	std::pair<Move, TreeNode<State>*> select(double c_puct) const;

	template<typename RandomEngine>
	std::pair<Move, TreeNode<State>*> env_select(RandomEngine*) const;

	void update_recursive(double leaf_value);
	double get_U_value(double c_puct) const;
	bool is_leaf() const;
	bool is_root() const;

private:
	void update(double leaf_value);

protected:
	TreeNode<State>* const _parent;
	std::vector<std::pair<Move, TreeNode<State>*>> _children;
	unsigned int _n_visit = 0;
	double _Q = 0;
	double _prior;
};

#include "TreeNode.ipp"

template<typename State>
class MCTS
{
public:
	typedef typename State::Move Move;

	typedef std::function<std::pair<std::vector<std::pair<typename State::Move, double>>, double>(const State&)> PolicyFunction;

	MCTS(const PolicyFunction& _policy_fn, double _c_puct, unsigned int _n_playout) :
		_root(new TreeNode<State>(nullptr, 1.0)),
		_current_root(_root),
		_policy_fn(_policy_fn),
		_c_puct(_c_puct),
		_n_playout(_n_playout)
	{ }

	~MCTS() { delete _root; }

	std::pair<std::vector<typename State::Move>, std::vector<double>> get_move_probs(State& state, bool small_temp=false);

    void update_with_move_index(State curState, unsigned int move_index);
    void update_with_move(const State& nextState, Move move);

    void reset();
private:

	template<typename RandomEngine>
	void _playout(State state, RandomEngine* rng);

	TreeNode<State>* _root;
	TreeNode<State>* _current_root;
	const PolicyFunction _policy_fn;
	double _c_puct;
	unsigned int _n_playout;
};

#include "mcts.ipp"

template<typename State>
class BatchMCTS
{
public:
	typedef typename State::Move Move;
	typedef std::pair<std::vector<std::pair<typename State::Move, double>>, double> EvalResult;

	typedef std::function<void(const std::vector<State>& boards, std::vector<EvalResult>&, int, void*)> PolicyFunction;

	BatchMCTS(const PolicyFunction& policy_fn, std::size_t compact_state_size, double c_puct, std::size_t n_playout, std::size_t thread_pool_size, std::size_t eval_batch_size);

	~BatchMCTS();

	std::vector<std::pair<std::vector<typename State::Move>, std::vector<double>>> get_move_probs(std::vector<State>& state, const std::vector<bool>& small_temp);

	void reset();
	
private:

	template<typename RandomEngine>
	TreeNode<State>* _playout_single_path(TreeNode<State>* root, State& state, std::vector<int>& players, double& leaf_value, bool& game_ended, RandomEngine* rng);

	template<typename RandomEngine>
	void _playout_batch(const std::vector<State>& state, std::size_t start_i, std::size_t n_games, RandomEngine* rng);

	void _backprop_single_path(TreeNode<State>* node, double leaf_value, const std::vector<int>& players);

	int _eval_and_backprop_batch(const std::vector<TreeNode<State>*>& nodes, 
		const std::vector<State>& states, 
		const std::vector<std::vector<int>>& players, 
		const std::vector<double>& batch_ended_results,
    	const std::vector<double>& compact_state_buffer,
		std::vector<BatchMCTS<State>::EvalResult>& eval_results, 
		int batch_size
	);

	std::vector<TreeNode<State>*> _roots;
	const PolicyFunction _policy_fn;
	std::size_t _compact_state_size;

	double _c_puct;
	std::size_t _n_playout;

	std::size_t _eval_batch_size; 
	std::size_t _thread_pool_size;

	threading::ThreadPool _pool;

	int _depth = 0;
};

#include "batch_mcts.ipp"

}
#endif