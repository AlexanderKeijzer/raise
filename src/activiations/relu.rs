use crate::tensor::Tensor;
use crate::layers::layer::Layer;
use crate::ops::ops;
pub struct ReLU {
    input: Tensor,
}

impl ReLU {
    pub fn new(size: usize) -> ReLU{
        ReLU {
            input: Tensor::zeros(&[0, size]),
        }
    }
}

impl Layer for ReLU {

    fn forward(&self, tensor: &Tensor) -> Tensor{
        ops::max(0., tensor)
    } 

    fn backward(&mut self, output: &Tensor) {
        self.input.gradient = Some(output.to_vec().iter().map(|x| ((x > &0.) as i32 as f32) ).collect())
    }

    fn get_input(&mut self) -> &mut Tensor {
        &mut self.input
    }

    fn set_input(&mut self, tensor: Tensor) {
        self.input = tensor;
    }
}