use super::loss::Loss;
use crate::tensor::Tensor;

pub struct MSE {
    input: Tensor,
}

impl MSE {
    pub fn new(size: usize) -> MSE{
        MSE {
            input: Tensor::zeros(&[1, size])
        }
    }
}

impl Loss for MSE {
    fn forward(&self, input: &Tensor, target: &Tensor) -> f32 {
        (input-target).pow(2.).mean()
    } 

    fn backward(&mut self, target: &Tensor) {
        self.get_input().gradient = Some(Box::new(&(2.*&(self.get_input()-target)) / (target.shape[0] as f32)));
    }

    fn get_input(&mut self) -> &mut Tensor {
        &mut self.input
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = tensor;
    }
}