use crate::tensor::Tensor;

pub struct DataSet {
    input: Tensor,
    target: Tensor
}

impl DataSet {
    pub fn new(input: Tensor, target: Tensor) -> DataSet {
        DataSet {
            input,
            target
        }
    }


}