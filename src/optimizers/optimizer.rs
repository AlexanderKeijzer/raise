use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&self, parameters: Vec<&mut Tensor>);
}