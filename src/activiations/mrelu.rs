use crate::tensor::Tensor;
use crate::layers::layer::Layer;

#[derive(Clone)]
#[allow(non_camel_case_types)]
pub struct mReLU {
    input: Option<Tensor>,
}

impl mReLU {
    pub fn new() -> mReLU {
        mReLU {
            input: None,
        }
    }
}

impl Layer for mReLU {

    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone().clamp_min(0.) - 0.5
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