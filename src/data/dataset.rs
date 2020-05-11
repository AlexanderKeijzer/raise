use crate::tensor::Tensor;

pub struct DataSet {
    pub input: Tensor,
    pub target: Tensor
}

impl DataSet {
    pub fn new(input: Tensor, target: Tensor) -> DataSet {
        assert!(input.shape[3] == target.shape[3]);
        DataSet {
            input,
            target
        }
    }

    pub fn len(&self) -> usize {
        self.input.shape[3]
    }

    pub fn get(&self, start: usize, end: usize) -> (Tensor, Tensor) {
        (self.input.get_minibatch(start, end), self.target.get_minibatch(start, end))
    }

    pub fn split(self, size_first: f32) -> (DataSet, DataSet) {
        let (inp_first, inp_second) = self.input.split(size_first);
        let (tar_first, tar_second) = self.target.split(size_first);
        (DataSet::new(inp_first, tar_first), DataSet::new(inp_second, tar_second))
    }
    
    pub fn norm_input(&mut self) {
        self.input.norm();
    }

    pub fn input_shape(&self) -> [usize; 3] {
        [self.input.shape[0], self.input.shape[1], self.input.shape[2]]
    }

    pub fn target_shape(&self) -> [usize; 3] {
        [self.target.shape[0], self.target.shape[1], self.target.shape[2]]
    }
}