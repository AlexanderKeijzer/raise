use crate::layers::layer::Layer;
use crate::tensor::Tensor;

pub struct Print {
    ident: String
}

impl Print {
    pub fn new(ident: &str) -> Print {
        Print {
            ident: String::from(ident)
        }
    }
}

impl Layer for Print {
    fn fwd(&mut self, tensor: Tensor) -> Tensor {
        println!("{}: {}", self.ident, tensor.mean_all());
        tensor
    }

    fn forward(&self, _: &Tensor) -> Tensor {
        Tensor::zeros([0, 0, 0, 0])
    }

    fn bwd(&mut self, tensor: Tensor) -> Tensor {
        tensor
    }

    fn backward(&mut self, _: Tensor, _: Tensor) -> Tensor {
        Tensor::zeros([0, 0, 0, 0])
    }

    fn take_input(&mut self) -> Tensor {
        Tensor::zeros([0, 0, 0, 0])
    }

    fn set_input(&mut self, _: Tensor) {
    }
}