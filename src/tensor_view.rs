/*
use crate::tensor::Tensor;

pub struct TensorView<'a> {
    pub shape: [usize; 4],
    values: &'a [f32],
    pub gradient: Option<Box<&'a Tensor>>
}

impl<'a> TensorView<'a> {
    pub fn new(tensor: &'a Tensor, index: usize) -> TensorView<'a> {
        let val_size = tensor.shape[0]*tensor.shape[1];
        TensorView {
            shape: tensor.shape,
            values: &(tensor.values)[index*val_size..(index*val_size + val_size)],
            gradient: None
        }
    }
}
*/