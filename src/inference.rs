pub trait Inference {}

#[derive(Default)]
pub struct NoInference;

impl Inference for NoInference {}
