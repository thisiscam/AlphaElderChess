#ifndef MCTS_H
#define MCTS_H

#include <stdexcept>
#include <sstream>
#include <random>
#include <iostream>

namespace mcts {

extern std::default_random_engine rng;

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

	void expand(std::vector<std::pair<Move, double>> priors);
	std::pair<Move, TreeNode<State>*> select(double c_puct) const;
	std::pair<Move, TreeNode<State>*> env_select() const;
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

	std::pair<std::vector<typename State::Move>, std::vector<double>> get_move_probs(State& state, double temp=1e-3);

    void update_with_move_index(State curState, unsigned int move_index);
    void update_with_move(const State& nextState, Move move);

    void reset();
private:

	void _playout(State state);

	TreeNode<State>* _root;
	TreeNode<State>* _current_root;
	const PolicyFunction _policy_fn;
	double _c_puct;
	unsigned int _n_playout;
};

#include "mcts.ipp"

}
#endif