use crate::tensor::Tensor;
use dyn_clone::DynClone;

pub trait Loss: DynClone {

    fn forward(&self, tensor: &Tensor, target: &Tensor) -> f32;

    fn fwd(&mut self, input: Tensor, target: &Tensor) -> f32 {
        let tmp = self.forward(&input, target);
        self.set_input(input);
        tmp
    }

    fn backward(&mut self, input: Tensor, target: Tensor) -> Tensor;

    fn bwd(&mut self, target: Tensor) -> Tensor {
        let input = self.take_input();
        self.backward(input, target)
    }

    fn take_input(&mut self) -> Tensor;

    fn set_input(&mut self, tensor: Tensor);
}