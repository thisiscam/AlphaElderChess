template<typename State>
template<typename RandomEngine>
void MCTS<State>::_playout(State state, RandomEngine* rng) {
	TreeNode<State>* node = _current_root;
	std::vector<int> players;
	while(true) {
		players.push_back(state.get_current_player());
		if(node->is_leaf()) {
			if(state.is_env_move()) {
				node->expand(state.get_env_move_weights());
				auto action_node = node->env_select(rng);
				node = action_node.second;
				state.do_move(action_node.first);
			} else {
				break;
			}
		} else {
			if(state.is_env_move()) {
				auto action_node = node->env_select(rng);
				node = action_node.second;
				state.do_move(action_node.first);
			} else {
				auto action_node = node->select(_c_puct);
				node = action_node.second;
				state.do_move(action_node.first);
			}
		}
	}
	double leaf_value;
	if(state.game_ended()) {
		auto winner = state.get_winner();
		if(winner == 2) {
			leaf_value = 0;
		} else if (winner == state.get_current_player()) {
			leaf_value = 1.;
		} else {
			assert(winner == 1 - state.get_current_player());
			leaf_value = -1.;
		}
	} else {
		auto policy_value_pair = this->_policy_fn(state);
		node->expand(policy_value_pair.first);
		leaf_value = policy_value_pair.second;
	}

	int last_player = state.get_current_player();
	TreeNode<State> *it = node;
	for(int i = players.size() - 2; i >= 0; i--) {
		int player = players[i];
		if(player == last_player) {
			it->update(leaf_value);
			leaf_value *= 0.99;
		} else if (player == 1 - last_player) {
			it->update(-leaf_value);
			leaf_value *= 0.99;
		} else {
			it->update(0.);
		}
		it = it->_parent;
	}
}

template<typename State>
std::pair<std::vector<typename State::Move>, std::vector<double>> MCTS<State>::get_move_probs(State& state, bool small_temp) {
	for(int i = 0; i < _n_playout; i++) {
		_playout(state, &rng);
	}
	if(small_temp) {
		std::vector<typename State::Move> moves(_current_root->_children.size());
		std::vector<double> counts(_current_root->_children.size());
		int max_c = -1;
		int max_idx = 0;
		for(int i = 0; i < _current_root->_children.size(); i++) {
			int c = _current_root->_children[i].second->_n_visit;
			moves[i] = _current_root->_children[i].first;
			if(c > max_c) {
				max_idx = i;
				max_c = c;
			}
		}
		counts[max_idx] = 1.;
		return std::make_pair(moves, counts);
	} else {
		double sum = 0.;
		std::vector<typename State::Move> moves(_current_root->_children.size());
		std::vector<double> counts(_current_root->_children.size());
		for(int i = 0; i < _current_root->_children.size(); i++) {
			double c = (double)_current_root->_children[i].second->_n_visit;
			counts[i] = c;
			moves[i] = _current_root->_children[i].first;
			sum += c;
		}
		for(int i = 0; i < _current_root->_children.size(); i++) {
			counts[i] /= sum;
		}
		return std::make_pair(moves, counts);
	}
}

template<typename State>
void MCTS<State>::update_with_move_index(State curState, unsigned int move_index) {
	curState.do_move(_current_root->_children[move_index].first);
	State& nextState = curState;
	TreeNode<State>* new_root = _current_root->_children[move_index].second;
	// _current_root->_children[move_index].second = nullptr;
	// delete _current_root;
	_current_root = new_root;
	if(nextState.is_env_move() && _current_root->is_leaf()) {
		_current_root->expand(nextState.get_env_move_weights());
	}
}

template<typename State>
void MCTS<State>::update_with_move(const State& nextState, typename State::Move move) {
	if(_current_root->is_leaf()) {
		// This can happen if we encountered a totally new state
		// we should be able to drop anything we've calculated before and move on
		return;
	}
	for(int i = 0; i < _current_root->_children.size(); i++) {
		if(_current_root->_children[i].first == move) {
			update_with_move_index(nextState, i);
			return;
		}
	}
	throw std::runtime_error("move not found");
}

template<typename State>
void MCTS<State>::reset() {
	delete _root;
	_root = new TreeNode<State>(nullptr, 1.0);
	_current_root = _root;
}
