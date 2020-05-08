use crate::data::dataset::DataSet;
use crate::tensor::Tensor;
use crate::data::sampler::*;

pub struct DataLoader {
    dataset: DataSet,
    batch_size: usize, 
    shuffle: bool
}

impl<'a> DataLoader {
    pub fn new(dataset: DataSet, batch_size: usize, shuffle: bool) -> DataLoader {
        DataLoader {
            dataset: dataset,
            batch_size: batch_size,
            shuffle: shuffle
        }
    }

    pub fn batches(&'a self) -> Box<dyn Iterator<Item = (Tensor, Tensor)> + 'a> {
        if self.shuffle {
            Box::new(RandomSampler::new(&self))
        } else {
            Box::new(OrderedSampler::new(&self))
        }
    }

    pub fn len(&self) -> usize {
        self.dataset.len()/self.batch_size
    }

    pub fn get(&self, batch_pos: usize) -> (Tensor, Tensor) {
        self.dataset.get(batch_pos*self.batch_size, (batch_pos+1)*self.batch_size)
    }
}