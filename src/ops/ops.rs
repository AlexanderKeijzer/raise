use crate::tensor::Tensor;

pub fn max(scalar: f32, tensor: &Tensor) -> Tensor {
    let mut values = tensor.to_vec();
    for i in 0..values.len() {
        if values[i] < scalar {
            values[i] = scalar;
        }
    }
    Tensor::new(values, tensor.shape)
}