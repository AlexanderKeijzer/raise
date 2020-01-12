use crate::tensor::Tensor;

pub trait Loss {

    fn forward(&self, tensor: &Tensor, target: &Tensor) -> f32;

    fn fwd(&mut self, input: Tensor, target: &Tensor) -> f32 {
        let tmp = self.forward(&input, target);
        self.set_input(input);
        tmp
    }

    fn backward(&mut self, target: &Tensor);

    fn bwd(&mut self, target: &Tensor) {
        self.backward(target);
    }

    fn get_input(&mut self) -> &mut Tensor;

    fn set_input(&mut self, tensor: Tensor);
}