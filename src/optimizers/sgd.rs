use crate::optimizers::optimizer::Optimizer;
use crate::tensor::Tensor;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> SGD{
        SGD {
            learning_rate: learning_rate
        }
    }
}

impl Optimizer for SGD {

    fn step(&self, mut parameters: Vec<&mut Tensor>) {
        for tensor in parameters.iter_mut() {
            *tensor -= &(self.learning_rate*tensor.gradient.as_ref().unwrap().as_ref().sum(3));
        }
    }
}