use crate::data::dataloader::DataLoader;
use crate::tensor::Tensor;

use rand::thread_rng;
use rand::seq::SliceRandom;

pub struct RandomSampler<'a> {
    dataloader: &'a DataLoader,
    next_batches: Vec<usize>,
}

impl<'a> RandomSampler<'a> {
    pub fn new(dataloader: &DataLoader) -> RandomSampler {
        let mut rng = thread_rng();
        let mut selection: Vec<usize> = (0..dataloader.len()).collect();
        selection.shuffle(&mut rng);

        RandomSampler {
            dataloader: dataloader,
            next_batches: selection
        }
    }
}

impl<'a> Iterator for RandomSampler<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<(Tensor, Tensor)> {
        match self.next_batches.pop() {
            Some(batch) => Some(self.dataloader.get(batch)),
            None => None

        }
    }
}

impl<'a> ExactSizeIterator for RandomSampler<'a> {
    fn len(&self) -> usize {
        self.dataloader.len()
    }
}

pub struct OrderedSampler<'a> {
    dataloader: &'a DataLoader,
    location: usize
}

impl<'a> OrderedSampler<'a> {
    pub fn new(dataloader: &DataLoader) -> OrderedSampler {

        OrderedSampler {
            dataloader: dataloader,
            location: 0
        }
    }
}

impl<'a> Iterator for OrderedSampler<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<(Tensor, Tensor)> {
        if self.location < self.len() {
            let res = self.dataloader.get(self.location);
            self.location += 1;
            Some(res)
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for OrderedSampler<'a> {
    fn len(&self) -> usize {
        self.dataloader.len()
    }
}