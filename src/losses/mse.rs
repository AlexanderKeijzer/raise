use super::loss::Loss;
use crate::tensor::Tensor;

#[derive(Clone)]
pub struct MSE {
    input: Option<Tensor>,
}

impl MSE {
    pub fn new() -> MSE {
        MSE {
            input: None
        }
    }
}

impl Loss for MSE {
    fn forward(&self, input: &Tensor, target: &Tensor) -> f32 {
        (input-target).pow(2.).mean_all()
    } 

    fn backward(&mut self, input: Tensor, target: Tensor) -> Tensor {
        let s = target.shape[1] as f32;
        2.*(input-target) / s
    }

    fn take_input(&mut self) -> Tensor {
        self.input.take().unwrap()
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = Some(tensor);
    }
}