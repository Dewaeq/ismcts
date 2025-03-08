use super::{action_list::ActionList, edge::Edge, node::Node, state::State};

pub struct Tree<T: State> {
    nodes: Vec<Node<T>>,
    index: usize,
    c: f32,
    default_capacity: usize,
}

impl<T> Tree<T>
where
    T: State + Clone,
{
    pub fn new(c: f32, default_capacity: usize) -> Self {
        Tree {
            nodes: Vec::with_capacity(default_capacity),
            index: 0,
            c,
            default_capacity,
        }
    }

    pub fn reset(&mut self) {
        self.index = 0;
        self.nodes = Vec::with_capacity(self.default_capacity);
    }

    pub fn add_node(
        &mut self,
        edge: Option<Edge<T::Action, usize>>,
        parent_id: Option<usize>,
    ) -> usize {
        let node_id = self.index;

        if let Some(parent_id) = parent_id {
            self.nodes[parent_id].add_child(node_id);
        }

        let node = Node::new(edge, parent_id);

        self.nodes.push(node);
        self.index += 1;

        node_id
    }

    pub fn select(&mut self, mut node_id: usize, state: &mut T) -> usize {
        let mut legal_actions = state.possible_actions();

        while !state.is_terminal() && self.is_fully_expanded(node_id, &legal_actions) {
            node_id = self.uct_select_child(node_id, &legal_actions).unwrap();

            let action = self.get_edge(node_id).unwrap().action();
            state.apply_action(action);

            if state.is_terminal() {
                break;
            }
            legal_actions = state.possible_actions();
        }

        node_id
    }

    /// this method relies on the fact that child ids are strictly greater than the
    /// id of their parent
    fn uct_select_child(&mut self, node_id: usize, legal_actions: &T::ActionList) -> Option<usize> {
        let mut best_child = None;
        let mut best_score = f32::MIN;

        let split_pos = node_id + 1;
        let (left, right) = self.nodes.split_at_mut(split_pos);
        let parent = left.last().unwrap();

        for &child_id in parent.child_ids_ref().iter() {
            let action = right[child_id - split_pos].edge().unwrap().action();
            if legal_actions.has(&action) {
                let uct_score = right[child_id - split_pos].uct_score(self.c);
                if uct_score > best_score {
                    best_score = uct_score;
                    best_child = Some(child_id);
                }
                right[child_id - split_pos].increase_availability();
            }
        }

        best_child
    }

    pub fn expand(&mut self, node_id: usize, state: &mut T) -> usize {
        if state.is_terminal() {
            return node_id;
        }

        let legal_actions = state.possible_actions();

        match self.nodes[node_id].pop_action(&legal_actions) {
            None => node_id,
            Some(action) => {
                let actor = state.turn();
                let edge = Edge::new(action.clone(), actor);

                state.apply_action(action);
                self.add_node(Some(edge), Some(node_id))
            }
        }
    }

    pub fn best_action(&self, node_id: usize, state: &T) -> Option<T::Action> {
        let legal_actions = state.possible_actions();
        let child_id = self.nodes[node_id]
            .child_ids_ref()
            .iter()
            .filter(|&&child_id| legal_actions.has(&self.get_edge(child_id).unwrap().action()))
            .max_by_key(|&&child_id| self.nodes[child_id].num_sims())
            .unwrap();

        self.get_edge(*child_id).map(|e| e.action())
    }

    pub fn scored_actions(&self, node_id: usize, state: &T) -> Vec<(f32, T::Action)> {
        let mut results = vec![];
        let legal_actions = state.possible_actions();

        for &child_id in self.nodes[node_id].child_ids_ref() {
            if let Some(edge) = self.get_edge(child_id) {
                let action = edge.action();
                if legal_actions.has(&action) {
                    results.push((self.nodes[child_id].uct_score(self.c), action))
                }
            }
        }

        results
    }

    pub fn update_node(&mut self, node_id: usize, reward: f32) {
        self.nodes[node_id].update(reward);
    }

    pub fn is_fully_expanded(&self, node_id: usize, legal_actions: &T::ActionList) -> bool {
        !self.nodes[node_id].has_untried_actions(legal_actions)
    }

    pub fn get_parent_id(&self, node_id: usize) -> Option<usize> {
        self.nodes[node_id].parent_id()
    }

    pub fn get_edge(&self, node_id: usize) -> Option<Edge<T::Action, usize>> {
        self.nodes[node_id].edge()
    }
}
