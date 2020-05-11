
use crate::tensor::Tensor;
use super::layer::Layer;

pub struct Linear {
    input: Option<Tensor>,
    pub weights: Tensor,
    biases: Tensor
}

impl Linear {
    pub fn new(shape: [usize; 2], init: &str) -> Linear {

        let weights;
        if init.eq("relu") {
            weights = Tensor::rand([shape[0], shape[1], 1, 1])*(2./(shape[0] as f32)).sqrt();
        } else if init.eq("pytorch") {
            let gain = (2_f32/(1_f32+5_f32.sqrt().powi(2))).sqrt();
            let std = gain/((shape[0] as f32).sqrt());
            let bound = 3_f32.sqrt()*std;
            weights = Tensor::uniform([shape[0], shape[1], 1, 1], -bound, bound)
        } else {
            weights = Tensor::rand([shape[0], shape[1], 1, 1])*(2./(shape[0] as f32)).sqrt();
        }

        let biases;
        if init.eq("pytorch") {
            let bound = 1_f32/(shape[0] as f32).sqrt();
            biases = Tensor::uniform([1, shape[1], 1, 1], -bound, bound)
        } else {
            biases = Tensor::zeros([1, shape[1], 1, 1]);
        }

        Linear {
            input: None,
            weights: weights,
            biases: biases
        }
    }
}

impl Layer for Linear {

    fn forward(&self, tensor: &Tensor) -> Tensor {
        &self.weights*tensor + &self.biases
    } 

    fn backward(&mut self, input: Tensor, output_grad: Tensor) -> Tensor {

        self.weights.gradient = Some(Box::new(output_grad.outermean3(&input))); // 68.5%
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