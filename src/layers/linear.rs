
use crate::tensor::Tensor;
use super::layer::Layer;

pub struct Linear {
    input: Tensor,
    weights: Tensor,
    biases: Tensor
}

impl Linear {
    pub fn new(shape: [usize; 2]) -> Linear{

        Linear {
            input: Tensor::zeros(&[shape[0], 1]),
            weights: Tensor::rand(&shape),
            biases: Tensor::zeros(&[shape[0], 1])
        }
    }
}

impl Layer for Linear {

    fn forward(&self, tensor: &Tensor) -> Tensor{
        &(&self.weights*tensor) + &self.biases
    } 

    fn backward(&mut self, output: &Tensor) {
        //Redo
        let grad = output.gradient.as_ref().unwrap();
        let l = grad.len();
        let v = vec![l, 1];
        let grad_tensor = Tensor::new(grad.clone(), vec![l, 1]);

        self.input.gradient = Some((&self.weights.T()*&grad_tensor).to_vec());
        self.weights.gradient = Some((&grad_tensor*&self.get_input().T()).to_vec());
        self.biases.gradient = Some(grad.clone());

    }

    fn get_input(&mut self) -> &mut Tensor {
        &mut self.input
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = tensor;
    }
}