use crate::layers::layer::Layer;
use crate::tensor::Tensor;
use std::ops::{Index, IndexMut};

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Sequential {
        Sequential {
            layers: layers
        }
    }
}

impl Layer for Sequential {
    fn fwd(&mut self, mut tensor: Tensor) -> Tensor {
        for layer in self.layers.iter_mut() {
            tensor = layer.as_mut().fwd(tensor);
        }
        tensor
    }

    fn forward(&self, _: &Tensor) -> Tensor {
        Tensor::zeros([0, 0, 0, 0])
    }

    fn bwd(&mut self, mut tensor: Tensor) -> Tensor {
        for layer in self.layers.iter_mut().rev() {
            tensor = layer.as_mut().bwd(tensor);
        }
        tensor
    }

    fn backward(&mut self, _: Tensor, _: Tensor) -> Tensor {
        Tensor::zeros([0, 0, 0, 0])
    }

    fn take_input(&mut self) -> Tensor {
        Tensor::zeros([0, 0, 0, 0])
    }

    fn set_input(&mut self, _: Tensor) {
    }

    fn get_parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in self.layers.iter_mut() {
            params.append(&mut layer.as_mut().get_parameters());
        }
        params
    }
}

impl Index<usize> for Sequential {
    type Output = Box<dyn Layer>;

    fn index(&self, index: usize) -> &Box<dyn Layer> {
        &self.layers[index]
    }
}

impl IndexMut<usize> for Sequential {
    fn index_mut(&mut self, index: usize) -> &mut Box<dyn Layer> {
        &mut self.layers[index]
    }

}