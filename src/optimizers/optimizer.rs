use crate::tensor::Tensor;
use dyn_clone::DynClone;

pub trait Optimizer: DynClone {
    fn step(&self, parameters: Vec<&mut Tensor>);

    fn set_learning_rate(&mut self, learning_rate: f32);
}