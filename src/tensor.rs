extern crate rand;
extern crate rand_distr;

use std::ops::{Index, IndexMut, Mul, Add, Sub, Div};
use std::fmt;
use std::fmt::Display;
use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    values: Vec<f32>,
    pub gradient: Option<Vec<f32>>
}

impl Tensor {
    pub fn new(values:Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor {
            shape: shape,
            values: values,
            gradient: None
        }
    }
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor {
            shape: shape.to_vec(),
            values: vec![0.; length_flat_indices(shape)],
            gradient: None
        }
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor {
            shape: shape.to_vec(),
            values: vec![1.; length_flat_indices(shape)],
            gradient: None
        }
    }

    pub fn rand(shape: &[usize]) -> Tensor {
        let mut rng = thread_rng();
        Tensor {
            shape: shape.to_vec(),
            values: (0..length_flat_indices(shape)).map(|_| rng.sample(StandardNormal)).collect(),
            gradient: None
        }
    }

    //Decide how to communicate inplace functions. Maybe like python with _?
    pub fn pow(&mut self, exponent: f32) -> &Tensor {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].powf(exponent);
        }
        self
    }

    pub fn mean(&self) -> f32 {
        (self.values.iter().sum::<f32>())/(self.values.len() as f32)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        //Make sure tensor is destroyed and move values instead of cloning?
        self.values.clone()
    }

    //TODO: redo on 1D vec, this seems very inefficent
    pub fn T(&self) -> Tensor {
        let mut t = self.clone();
        let new_shape = [t.shape[1], t.shape[0]];
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                t.values.swap(flatten_indices(&self, [i, j]), flatten_indices2(new_shape, [j, i]));
            }
        }
        t.shape = new_shape.to_vec();
        t
    }
}

fn length_flat_indices(indices: &[usize]) -> usize {
    let mut u = 1;
    for i in 0..indices.len() {
        u *= indices[i];
    }
    u
}

fn flatten_indices(tensor: &Tensor, indices: [usize; 2]) -> usize {
    assert!(tensor.shape.len() == indices.len());

    let mut index = 0;
    let mut mul = 1;
    
    for i in 0..tensor.shape.len() {
        index += indices[i]*mul;
        mul *= tensor.shape[i];
    }
    index
}

fn flatten_indices2(shape: [usize; 2], indices: [usize; 2]) -> usize {
    assert!(shape.len() == indices.len());

    let mut index = 0;
    let mut mul = 1;
    
    for i in 0..shape.len() {
        index += indices[i]*mul;
        mul *= shape[i];
    }
    index
}

/*
fn flatten_indices(tensor: &Tensor, indices: &[usize]) -> usize {
    assert!(tensor.shape.len() == indices.len());

    let index = 0;
    let mul = 1;
    
    for i in 0..tensor.shape.len() {
        index += indices[i]*mul;
        mul *= tensor.shape[i];
    }

    index
}
*/

fn reshape_index(tensor: &Tensor, mut index: usize) -> Vec<usize> {
    assert!(index < tensor.values.len());

    let mut indices = Vec::new();

    let mut mul = tensor.values.len();
    
    for i in (0..tensor.shape.len()).rev() {
        mul /= tensor.shape[i-1];
        indices[i-1] = index / mul;
        index -= indices[i - 1] * mul;
    }

    indices
}

/*
impl Index<&[usize]> for Tensor {
    type Output = f32;

    fn index(&self, indices: &[usize]) -> &f32 {
        &self.values[flatten_indices(self, indices)]
    }
}

impl IndexMut<&[usize]> for Tensor {

    fn index_mut(&mut self, indices: &[usize]) -> &mut f32 {
        &mut self.values[flatten_indices(self, indices)]
    }
}
*/

impl Index<[usize; 2]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 2]) -> &f32 {
        &self.values[flatten_indices(&self, indices)]
    }
}

impl IndexMut<[usize; 2]> for Tensor {

    fn index_mut(&mut self, indices: [usize; 2]) -> &mut f32 {
        let index;
        {
            index = flatten_indices(&self, indices);
        }
        &mut self.values[index]
    }
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        assert!(self.shape.len() == 1);

        &self.values[index]
    }
}

impl IndexMut<usize> for Tensor {

    fn index_mut(&mut self, index: usize) -> &mut f32 {
        assert!(self.shape.len() == 1);

        &mut self.values[index]
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[1]==rhs.shape[0]);

        let mut t = Tensor::zeros(&[self.shape[0], rhs.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                let mut val = 0.;
                for k in 0..self.shape[1] { 
                    val += self[[i, k]]*rhs[[k, j]];
                }
                t[[i, j]] = val;
            }
        }
        t
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {

        let mut t = self.clone();
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                t[[i, j]] *= rhs;
            }
        }
        t
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {

        let mut t = rhs.clone();
        for i in 0..rhs.shape[0] {
            for j in 0..rhs.shape[1] {
                t[[i, j]] *= self;
            }
        }
        t
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Tensor {

        let mut t = self.clone();
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                t[[i, j]] /= rhs;
            }
        }
        t
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0]==rhs.shape[0] && self.shape[1] ==rhs.shape[1]);

        let mut t = Tensor::zeros(&[self.shape[0], self.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                t[[i, j]] = self[[i, j]] + rhs[[i, j]]
            }
        }
        t
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0]==rhs.shape[0] && self.shape[1] ==rhs.shape[1]);

        let mut t = Tensor::zeros(&[self.shape[0], self.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                t[[i, j]] = self[[i, j]] - rhs[[i, j]]
            }
        }
        t
    }
}

//Is this REALLY necessary?
impl Sub<&Tensor> for &mut Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0]==rhs.shape[0] && self.shape[1] ==rhs.shape[1]);

        let mut t = Tensor::zeros(&[self.shape[0], self.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                t[[i, j]] = self[[i, j]] - rhs[[i, j]]
            }
        }
        t
    }
}


impl Display for Tensor {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "Tensor with shape: {:?}\n", self.shape)?;
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                write!(f, "{} ", self[[i, j]])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    } 
}