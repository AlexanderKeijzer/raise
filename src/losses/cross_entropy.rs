use super::loss::Loss;
use crate::tensor::Tensor;

pub struct CrossEntropy {
    input: Option<Tensor>,
}

impl CrossEntropy {
    pub fn new(size: usize) -> CrossEntropy {
        CrossEntropy {
            input: None
        }
    }
}

impl Loss for CrossEntropy {
    fn forward(&self, input: &Tensor, target: &Tensor) -> f32 {
        // Calculate the batch mean of the log_softmax prediction of the target
        // e.g. if target [0 1 0]T and log_softmax(input) = [0.2 0.4 0.3]T result should be 0.4
        //log_softmax(input);
        //TEMP MSE!!!
        (input-target).pow(2.).mean_all()
    } 

    fn backward(&mut self, input: Tensor, target: Tensor) -> Tensor {
        input-target
    }

    fn take_input(&mut self) -> Tensor {
        self.input.take().unwrap()
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = Some(tensor);
    }
}

pub fn log_softmax(tensor: &Tensor) -> Tensor {
    tensor - tensor.logsumexp()
}