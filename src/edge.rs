#[derive(Clone)]
pub struct Edge<A, B> {
    action: A,
    actor: B,
}

impl<A, B> Edge<A, B>
where
    A: Clone,
    B: Clone,
{
    pub fn new(action: A, actor: B) -> Self {
        Self { action, actor }
    }

    pub fn action(&self) -> A {
        self.action.clone()
    }

    pub fn actor(&self) -> B {
        self.actor.clone()
    }
}
