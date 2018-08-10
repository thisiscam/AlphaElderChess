template<typename State>
MCTS<State>::MCTS(const PolicyFunction& policy_fn, std::size_t compact_state_size, double c_puct, std::size_t n_playout, std::size_t thread_pool_size, std::size_t eval_batch_size) :
    _root(new TreeNode<State>(nullptr, 1.0)),
    _current_root(_root),
    _policy_fn(policy_fn),
    _compact_state_size(compact_state_size),
    _c_puct(c_puct),
    _n_playout(n_playout),
    _eval_batch_size(eval_batch_size),
    _thread_pool_size(thread_pool_size)
{
    _pool.initialize(thread_pool_size);
}

template<typename State>
MCTS<State>::~MCTS() { 
    delete _root; 
}

template<typename State>
template<typename RandomEngine>
TreeNode<State>* MCTS<State>::_playout_single_path(State& state, std::vector<int>& players, double& leaf_value, bool& game_ended, RandomEngine* rng) {
    TreeNode<State>* node = _current_root;
    while(true) {
        players.push_back(state.get_current_player());
        bool is_env_move = state.is_env_move();
        node->add_virtual_loss();
        node->lock();
        if(node->is_leaf()) {
            if(is_env_move) {
                node->expand(state.get_env_move_weights());
                node->unlock();
                auto action_node = node->env_select(rng);
                node = action_node.second;
                state.do_move(action_node.first);
            } else {
                node->unlock();
                break;
            }
        } else {
            node->unlock();
            if(is_env_move) {
                auto action_node = node->env_select(rng);
                node = action_node.second;
                state.do_move(action_node.first);
            } else {
                auto action_node = node->select(_c_puct, rng);
                node = action_node.second;
                state.do_move(action_node.first);
            }
        }
    }

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
        game_ended = true;
        return node;
    } else {
        game_ended = false;
        return node;
    }
}

template<typename State>
template<typename RandomEngine>
void MCTS<State>::_playout_batch(const State& state, std::size_t n_playout, RandomEngine* rng) {
    std::vector<State> batch_states(_eval_batch_size);
    std::vector<TreeNode<State>*> batch_nodes(_eval_batch_size);
    std::vector<std::vector<int>> batch_players(_eval_batch_size);

    std::vector<MCTS<State>::EvalResult> batch_eval_results(_eval_batch_size);
    std::vector<double> batch_ended_results(_eval_batch_size);

    std::vector<double> compact_state_buffer(_compact_state_size * _eval_batch_size);

    int total_ended_count = 0;

    int eval_count = 0;
    int ended_count = 0; // Trick: ended games are stored reverse in the above buffers

    int nn_eval_count = 0;

    for(int i = 0; i < n_playout; i++) {
        batch_states[eval_count] = state;
        std::vector<int> players;
        double leaf_value = 0.;
        bool game_ended = false;
        TreeNode<State>* node = _playout_single_path(batch_states[eval_count], players, leaf_value, game_ended, rng);
        
        int idx;
        if(game_ended) {
            if(eval_count == 0) {
                // can treat this as single playout
                assert(ended_count == 0);
                _backprop_single_path(node, leaf_value, players);
                total_ended_count++;
                continue;
            } else {
                idx = _eval_batch_size - ended_count - 1;
                batch_ended_results[idx] = leaf_value;
                ended_count++;
                total_ended_count++;
            }
        } else {
            idx = eval_count;
            eval_count++;
        }

        batch_nodes[idx] = node;
        batch_players[idx] = std::move(players);

        if(eval_count + ended_count == _eval_batch_size) {
            nn_eval_count += _eval_and_backprop_batch(batch_nodes, batch_states, batch_players, batch_ended_results, compact_state_buffer, batch_eval_results, eval_count);
            eval_count = 0;
            ended_count = 0;
        }
    }
    /* Backprop any residuals */
    if(eval_count + ended_count > 0) {
        nn_eval_count += _eval_and_backprop_batch(batch_nodes, batch_states, batch_players, batch_ended_results, compact_state_buffer, batch_eval_results, eval_count);
    }
    // std::cout << "ok" << total_ended_count << " " << nn_eval_count << " " << total_ended_count + nn_eval_count << std::endl;
}

template<typename State>
int MCTS<State>::_eval_and_backprop_batch(
    const std::vector<TreeNode<State>*>& nodes, 
    const std::vector<State>& states, 
    const std::vector<std::vector<int>>& players, 
    const std::vector<double>& batch_ended_results,
    const std::vector<double>& compact_state_buffer,
    std::vector<MCTS<State>::EvalResult>& eval_results,
    int eval_count) 
{
    int valid_cnt = 0;
    this->_policy_fn(states, eval_results, eval_count, (void*)compact_state_buffer.data());
    for(int i = 0; i < eval_count; i++) {
        TreeNode<State>* node = nodes[i];
        auto&& policy_value_pair = eval_results[i];
        bool do_backprop = false;
        node->lock();
        if(node->is_leaf()) {
            node->expand(policy_value_pair.first);
            do_backprop = true;
            valid_cnt ++;
        }
        node->unlock();
        if(do_backprop) {
            double leaf_value = policy_value_pair.second;
            _backprop_single_path(node, leaf_value, players[i]);
        } else {
            _remove_virtual_losses(node);
        }
    }
    for(int i = eval_count; i < _eval_batch_size; i++) {
        _backprop_single_path(nodes[i], batch_ended_results[i], players[i]);
    }
    return valid_cnt;
}

template<typename State>
void MCTS<State>::_backprop_single_path(TreeNode<State>* node, double leaf_value, const std::vector<int>& players) {
    int last_player = players[players.size() - 1];
    TreeNode<State> *node_back_it = node;
    for(int i = players.size() - 2; i >= 0; i--) {
        node_back_it->remove_virtual_loss();
        int player = players[i];
        if(player == last_player) {
            node_back_it->update(leaf_value);
        } else if (player == 1 - last_player) {
            node_back_it->update(-leaf_value);
        } else {
            node_back_it->update(0.);
        }
        node_back_it = node_back_it->_parent;
    }
    node_back_it->_n_visit++;
    node_back_it->remove_virtual_loss();
    assert(node_back_it == _current_root);
    // _current_root->debug_check_n_visit();
}

template<typename State>
void MCTS<State>::_remove_virtual_losses(TreeNode<State>* node) {
    do {
        node->remove_virtual_loss();
        node = node->_parent;
    } while(node != _current_root->_parent);
}


template<typename State>
std::pair<std::vector<typename State::Move>, std::vector<std::uint16_t>> MCTS<State>::get_move_counts(State& state) {

    threading::ThreadGroup tg(_pool);
    for(int i = 0; i < _thread_pool_size; i++) {
        std::size_t n_playout = _n_playout / _thread_pool_size + (i < (_n_playout % _thread_pool_size) ? 1 : 0);
        tg.add_task([this, i, n_playout, &state]() {
            std::mt19937 rng(time(0) + i); 
            this->_playout_batch(state, n_playout, &rng); 
        });
    }
    tg.wait_all();

    std::vector<typename State::Move> moves(_current_root->_children.size());
    std::vector<std::uint16_t> counts(_current_root->_children.size());
    for(int i = 0; i < _current_root->_children.size(); i++) {
        std::uint16_t c = _current_root->_children[i].second->_n_visit;
        counts[i] = c;
        moves[i] = _current_root->_children[i].first;
    }
    return std::make_pair(moves, counts);
}

template<typename State>
void MCTS<State>::update_with_move_index(State curState, std::uint16_t move_index) {
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
