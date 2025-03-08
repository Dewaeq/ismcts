use std::time::Instant;

use super::{state::State, tree::Tree};

pub struct SearchResult<T: State> {
    pub best_action: Option<T::Action>,
    pub scored_actions: Vec<(f32, T::Action)>,
}

pub struct Searcher<T: State + Clone> {
    tree: Tree<T>,
}

impl<T: State + Clone> Searcher<T> {
    pub fn new(c: f32, default_capacity: usize) -> Self {
        Searcher {
            tree: Tree::new(c, default_capacity),
        }
    }

    pub fn search(&mut self, state: &T, time: u128) -> SearchResult<T> {
        self.tree.reset();
        let root_id = self.tree.add_node(None, None);

        let mut i = 0;
        let started = Instant::now();

        loop {
            if i % 2048 == 0 && started.elapsed().as_millis() > time {
                break;
            }

            let mut state = state.randomize(state.turn());

            let node_id = self.tree.select(root_id, &mut state);
            let node_id = self.tree.expand(node_id, &mut state);
            state.do_rollout();
            self.backpropagate(&state, node_id);

            i += 1;
        }

        SearchResult {
            best_action: self.tree.best_action(root_id, state),
            scored_actions: self.tree.scored_actions(root_id, state),
        }
    }

    fn backpropagate(&mut self, state: &T, node_id: usize) {
        let mut node_id = Some(node_id);

        while let Some(id) = node_id {
            if let Some(edge) = self.tree.get_edge(id) {
                self.tree.update_node(id, state.reward(edge.actor()));
            }
            node_id = self.tree.get_parent_id(id);
        }
    }
}

impl<T: State + Clone> Default for Searcher<T> {
    fn default() -> Self {
        Self {
            tree: Tree::new(2f32.sqrt(), 500_000),
        }
    }
}
