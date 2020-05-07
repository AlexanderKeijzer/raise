
use crate::tensor::Tensor;
use super::layer::Layer;

pub struct Linear {
    input: Tensor,
    weights: Tensor,
    biases: Tensor
}

impl Linear {
    pub fn new(shape: [usize; 2]) -> Linear {

        Linear {
            input: Tensor::zeros([1, shape[1], 1, 1]),
            weights: Tensor::rand([shape[0], shape[1], 1, 1])/((shape[1] as f32).sqrt()), // He/Kaiming initialization?
            biases: Tensor::zeros([1, shape[1], 1, 1])
        }
    }
}

impl Layer for Linear {

    fn forward(&self, tensor: &Tensor) -> Tensor {
        &self.weights*tensor + &self.biases
    } 

    fn backward(&mut self, output: &Tensor) {
        let grad = output.gradient.as_ref().unwrap().as_ref();

        self.input.gradient = Some(Box::new(&self.weights.transpose()*grad)); // 19.1%
        self.weights.gradient = Some(Box::new((grad*&self.get_input().transpose()).mean(3))); // 68.5%
        self.biases.gradient = Some(Box::new(grad.mean(3))); // <0.1%
    }

    fn get_parameters(&mut self) -> Vec<&mut Tensor> {
        vec!(&mut self.weights, &mut self.biases)
    }

    fn get_input(&mut self) -> &mut Tensor {
        &mut self.input
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = tensor;
    }
}