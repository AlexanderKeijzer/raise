use crate::tensor::Tensor;

pub trait Layer {

    fn forward(&self, tensor: &Tensor) -> Tensor;

    fn fwd(&mut self, tensor: Tensor) -> Tensor{
        let tmp = self.forward(&tensor);
        self.set_input(tensor);
        tmp
    }

    fn backward(&mut self, output: &Tensor);

    fn bwd(&mut self, output: &Tensor) {
        self.backward(output);
    }

    fn get_parameters(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    fn get_input(&mut self) -> &mut Tensor;

    fn set_input(&mut self, tensor: Tensor);
}