template<typename State>
BatchMCTS<State>::BatchMCTS(const PolicyFunction& policy_fn, std::size_t compact_state_size, double c_puct, std::size_t n_playout, std::size_t thread_pool_size, std::size_t eval_batch_size) :
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
BatchMCTS<State>::~BatchMCTS() {
    for(auto _root : _roots) {
        delete _root;
    }
}

template<typename State>
template<typename RandomEngine>
TreeNode<State>* BatchMCTS<State>::_playout_single_path(TreeNode<State>* root, State& state, std::vector<int>& players, double& leaf_value, bool& game_ended, RandomEngine* rng) {
    TreeNode<State>* node = root;
    while(true) {
        players.push_back(state.get_current_player());
        bool is_env_move = state.is_env_move();
        if(node->is_leaf()) {
            if(is_env_move) {
                node->expand(state.get_env_move_weights());
                auto action_node = node->env_select(rng);
                node = action_node.second;
                state.do_move(action_node.first);
            } else {
                break;
            }
        } else {
            if(is_env_move) {
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
void BatchMCTS<State>::_playout_batch(
    const std::vector<State>& states, 
    std::size_t start_i, std::size_t n_games, 
    RandomEngine* rng) 
{
    std::vector<State> batch_states(_eval_batch_size);
    std::vector<TreeNode<State>*> batch_nodes(_eval_batch_size);
    std::vector<std::vector<int>> batch_players(_eval_batch_size);

    std::vector<BatchMCTS<State>::EvalResult> batch_eval_results(_eval_batch_size);
    std::vector<double> batch_ended_results(_eval_batch_size);

    std::vector<double> compact_state_buffer(_compact_state_size * _eval_batch_size);

    for(int j = 0; j < _n_playout; j++) {

        int total_ended_count = 0;

        int eval_count = 0;
        int ended_count = 0; // Trick: ended games are stored reverse in the above buffers

        int nn_eval_count = 0;
        
        for(int i = 0; i < n_games; i++) {
            int which_game = start_i + i;
            batch_states[eval_count] = State(states[which_game]);
            std::vector<int> players;
            double leaf_value = 0.;
            bool game_ended = false;
            TreeNode<State>* node = _playout_single_path(_roots[which_game], batch_states[eval_count], players, leaf_value, game_ended, rng);
            
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

        std::cout << "ok " << j << " " << _n_playout << " " << total_ended_count << " " << nn_eval_count << " " << total_ended_count + nn_eval_count << std::endl;
    }
}

template<typename State>
int BatchMCTS<State>::_eval_and_backprop_batch(
    const std::vector<TreeNode<State>*>& nodes, 
    const std::vector<State>& states, 
    const std::vector<std::vector<int>>& players, 
    const std::vector<double>& batch_ended_results,
    const std::vector<double>& compact_state_buffer,
    std::vector<BatchMCTS<State>::EvalResult>& eval_results,
    int eval_count) 
{
    int valid_cnt = 0;
    this->_policy_fn(states, eval_results, eval_count, (void*)compact_state_buffer.data());
    for(int i = 0; i < eval_count; i++) {
        TreeNode<State>* node = nodes[i];
        auto&& policy_value_pair = eval_results[i];
        bool do_backprop = false;
        if(node->is_leaf()) {
            node->expand(policy_value_pair.first);
            do_backprop = true;
            valid_cnt ++;
        }
        if(do_backprop) {
            double leaf_value = policy_value_pair.second;
            _backprop_single_path(node, leaf_value, players[i]);
        }
    }
    for(int i = eval_count; i < _eval_batch_size; i++) {
        _backprop_single_path(nodes[i], batch_ended_results[i], players[i]);
    }
    return valid_cnt;
}

template<typename State>
void BatchMCTS<State>::_backprop_single_path(TreeNode<State>* node, double leaf_value, const std::vector<int>& players) {
    int last_player = players[players.size() - 1];
    TreeNode<State> *node_back_it = node;
    for(int i = players.size() - 2; i >= 0; i--) {
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
    assert(node_back_it->_parent == nullptr);
}

template<typename State>
std::vector<std::pair<std::vector<typename State::Move>, std::vector<double>>> 
BatchMCTS<State>::get_move_probs(std::vector<State>& states, const std::vector<bool>& small_temp) {

    _roots.resize(states.size());
    for(int i = 0; i < states.size(); i++) {
        _roots[i] = new TreeNode<State>(nullptr, 1.0);
    }

    threading::ThreadGroup tg(_pool);
    for(int i = 0; i < _thread_pool_size; i++) {
        int residual = states.size() % _thread_pool_size;
        std::size_t num_states_this_thread = states.size() / _thread_pool_size + (i < residual ? 1 : 0);
        std::size_t start_i = i * (states.size() / _thread_pool_size) + std::min(i, residual);
        tg.add_task([this, i, start_i, num_states_this_thread, &states]() {
            std::mt19937 rng(time(0) + i); 
            this->_playout_batch(states, start_i, num_states_this_thread, &rng); 
        });
    }
    tg.wait_all();

    std::vector<std::pair<std::vector<typename State::Move>, std::vector<double>>> ret;
    for(int i = 0; i < states.size(); i++) {
        TreeNode<State> *root = _roots[i];
        if(small_temp[i]) {
            std::vector<typename State::Move> moves(root->_children.size());
            std::vector<double> counts(root->_children.size());
            int max_c = -1;
            int max_idx = 0;
            for(int i = 0; i < root->_children.size(); i++) {
                int c = root->_children[i].second->_n_visit;
                moves[i] = root->_children[i].first;
                if(c > max_c) {
                    max_idx = i;
                    max_c = c;
                }
            }
            counts[max_idx] = 1.;
            ret.push_back(std::make_pair(moves, counts));
        } else {
            double sum = 0.;
            std::vector<typename State::Move> moves(root->_children.size());
            std::vector<double> counts(root->_children.size());
            for(int i = 0; i < root->_children.size(); i++) {
                double c = (double)root->_children[i].second->_n_visit;
                counts[i] = c;
                moves[i] = root->_children[i].first;
                sum += c;
            }
            for(int i = 0; i < root->_children.size(); i++) {
                counts[i] /= sum;
            }
            ret.push_back(std::make_pair(moves, counts));
        }
    }
    return ret;
}

template<typename State>
void BatchMCTS<State>::reset() {
    for(auto _root : _roots) {
        delete _root;
    }
    _roots.clear();
}

