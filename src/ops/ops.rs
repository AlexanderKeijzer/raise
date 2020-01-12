use crate::tensor::Tensor;

pub fn max(scalar: f32, tensor: &Tensor) -> Tensor{
    let mut t = tensor.clone();
    for i in 0..t.shape[0] {
        for j in 0..t.shape[1] {
            if t[[i, j]] > scalar {
                t[[i, j ]] = scalar;
            }
        }
    }
    t
}