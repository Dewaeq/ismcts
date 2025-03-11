use crate::inference::Inference;

use super::action_list::ActionList;

pub trait State {
    type Action: Clone + PartialEq;
    type ActionList: ActionList<Self::Action> + Clone;
    type Inference: Inference;

    fn turn(&self) -> usize;

    fn randomize(&self, observer: usize, inference: &Self::Inference) -> Self;

    fn possible_actions(&self) -> Self::ActionList;

    fn next_state(&self, action: Self::Action) -> Self
    where
        Self: Clone,
    {
        let mut next_state = self.clone();
        next_state.apply_action(action);
        next_state
    }

    fn apply_action(&mut self, action: Self::Action);

    fn do_rollout(&mut self) {
        while !self.is_terminal() {
            let action = self.possible_actions().pop_random().unwrap();
            self.apply_action(action);
        }
    }

    fn is_terminal(&self) -> bool;

    fn reward(&self, perspective: usize) -> f32;
}
