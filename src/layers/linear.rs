
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
        //use std::time::Instant;

        let grad = output.gradient.as_ref().unwrap().as_ref();
        //println!("Output grad: {:?}", grad.shape);
        //println!("Input: {:?}", self.input.shape);
        //println!("{}", mem::size_of_val(grad));

        self.input.gradient = Some(Box::new(&self.weights.transpose()*grad));
        //println!("Input grad: {:?}", self.input.gradient.as_ref().unwrap().as_ref().shape);
        self.weights.gradient = Some(Box::new((grad*&self.get_input().transpose()).mean(3)));
        //println!("Weight grad: {:?}", self.weights.gradient.as_ref().unwrap().as_ref().shape);
        self.biases.gradient = Some(Box::new(grad.mean(3)));
        //println!("Bias grad: {:?}", self.biases.gradient.as_ref().unwrap().as_ref().shape);
        //panic!();
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