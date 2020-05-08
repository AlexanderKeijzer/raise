use crate::tensor::Tensor;

pub trait Layer {

    fn forward(&self, input: &Tensor) -> Tensor;

    fn fwd(&mut self, input: Tensor) -> Tensor {
        let tmp = self.forward(&input);
        self.set_input(input);
        tmp
    }

    fn backward(&mut self, input: Tensor, output_grad: Tensor) -> Tensor;

    fn bwd(&mut self, output_grad: Tensor) -> Tensor {
        let input = self.take_input();
        self.backward(input, output_grad)
    }

    fn get_parameters(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    fn take_input(&mut self) -> Tensor;

    fn set_input(&mut self, tensor: Tensor);
}