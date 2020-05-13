use crate::tensor::Tensor;
use crate::layers::layer::Layer;

#[derive(Clone)]
pub struct ReLU {
    input: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {
            input: None,
        }
    }
}

impl Layer for ReLU {

    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone().clamp_min(0.)
    } 

    fn backward(&mut self, input: Tensor, output_grad: Tensor) -> Tensor {
        &input.is_bigger_(0.)*&output_grad
    }

    fn take_input(&mut self) -> Tensor {
        self.input.take().unwrap()
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = Some(tensor);
    }
}