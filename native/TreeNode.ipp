template<typename State>
TreeNode<State>::~TreeNode() {
	for(auto it : _children) {
		delete it.second;
	}
}

template<typename State>
void TreeNode<State>::expand(const std::vector<std::pair<typename State::Move, double>>& priors) {
	for(auto& it : priors) {
		_children.push_back(std::make_pair(it.first, new TreeNode(this, it.second)));
	}
}

template<typename State>
std::pair<typename State::Move, TreeNode<State>*> TreeNode<State>::select(double c_puct) const {
	return *std::max_element(_children.begin(), _children.end(),
		[c_puct, this](auto a, auto b) { return a.second->get_U_value(c_puct) < b.second->get_U_value(c_puct); });
}

template<typename State>
template<typename RandomEngine>
std::pair<typename State::Move, TreeNode<State>*> TreeNode<State>::env_select(RandomEngine* rng) const {
	std::vector<double> weights;
	std::transform(
		_children.begin(), _children.end(), std::back_inserter(weights), 
		[](auto it) { return it.second->_prior; }
	);
	std::discrete_distribution<int> dist(std::begin(weights), std::end(weights));
	return _children[dist(*rng)];
}

template<typename State>
void TreeNode<State>::update(double leaf_value) {
	_n_visit ++;
	threading::atomic_add(_W, leaf_value);
}

template<typename State>
double TreeNode<State>::get_U_value(double c_puct) const {
	return (_W + c_puct * _prior * sqrt((double)_parent->_n_visit) - _virtual_loss) / (1 + _n_visit + _virtual_loss);
}

template<typename State>
bool TreeNode<State>::is_leaf() const {
	return _children.size() == 0;
}

template<typename State>
bool TreeNode<State>::is_root() const {
	return _parent == nullptr;
}

template<typename State>
void TreeNode<State>::lock() {
	_lock.lock();
}


template<typename State>
void TreeNode<State>::unlock() {
	_lock.unlock();
}

template<typename State>
void TreeNode<State>::add_virtual_loss() {
	_virtual_loss++;
}

template<typename State>
void TreeNode<State>::remove_virtual_loss() {
	_virtual_loss--;
}
