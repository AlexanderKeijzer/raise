use super::loss::Loss;
use crate::tensor::Tensor;

pub struct CrossEntropy {
    input: Tensor,
}

impl CrossEntropy {
    pub fn new(size: usize) -> CrossEntropy {
        CrossEntropy {
            input: Tensor::zeros([1, size, 1, 1])
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

    fn backward(&mut self, target: &Tensor) {
        self.input.gradient = Some(Box::new(&self.input-target));
    }

    fn get_input(&mut self) -> &mut Tensor {
        &mut self.input
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = tensor;
    }
}

pub fn log_softmax(tensor: &Tensor) -> Tensor {
    tensor - tensor.logsumexp()
}