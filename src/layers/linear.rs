
use crate::tensor::Tensor;
use super::layer::Layer;

pub struct Linear {
    input: Option<Tensor>,
    pub weights: Tensor,
    biases: Tensor
}

impl Linear {
    pub fn new(shape: [usize; 2], init: &str) -> Linear {

        let scalar;
        if init.eq("relu") {
            scalar = (2./(shape[0] as f32)).sqrt();
        } else {
            scalar = (1./(shape[0] as f32)).sqrt();
        }

        Linear {
            input: None,
            weights: Tensor::rand([shape[0], shape[1], 1, 1])*scalar, // He/Kaiming initialization?
            biases: Tensor::zeros([1, shape[1], 1, 1])
        }
    }
}

impl Layer for Linear {

    fn forward(&self, tensor: &Tensor) -> Tensor {
        &self.weights*tensor + &self.biases
    } 

    fn backward(&mut self, input: Tensor, output_grad: Tensor) -> Tensor {

        self.weights.gradient = Some(Box::new((&output_grad*&input.transpose()).mean(3))); // 68.5%
        self.biases.gradient = Some(Box::new(output_grad.mean(3))); // <0.1%

        &self.weights.transpose()*&output_grad // 19.1%
    }

    fn get_parameters(&mut self) -> Vec<&mut Tensor> {
        vec!(&mut self.weights, &mut self.biases)
    }

    fn take_input(&mut self) -> Tensor {
        self.input.take().unwrap()
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = Some(tensor);
    }
}