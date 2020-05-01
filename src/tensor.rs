extern crate rand;
extern crate rand_distr;

use std::ops::{Index, IndexMut, Mul, Add, Sub, Div, SubAssign};
use std::fmt;
use std::fmt::Display;
use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: [usize; 4],
    values: Vec<f32>,
    pub gradient: Option<Box<Tensor>>
}

// x y c b
impl Tensor {
    pub fn new(values:Vec<f32>, shape: [usize; 4]) -> Tensor {
        assert!(values.len() == length_flat_indices(shape));
        Tensor {
            shape: shape,
            values: values,
            gradient: None
        }
    }
    //pub fn new(values:Vec<f32>, shape: [usize; 3]) -> Tensor {Tensor::new(values, [shape, 1])}

    pub fn zeros(shape: [usize; 4]) -> Tensor {
        Tensor {
            shape: shape,
            values: vec![0.; length_flat_indices(shape)],
            gradient: None
        }
    }

    pub fn ones(shape: [usize; 4]) -> Tensor {
        Tensor {
            shape: shape,
            values: vec![1.; length_flat_indices(shape)],
            gradient: None
        }
    }

    pub fn rand(shape: [usize; 4]) -> Tensor {
        let mut rng = thread_rng();
        Tensor {
            shape: shape,
            values: (0..length_flat_indices(shape)).map(|_| rng.sample(StandardNormal)).collect(),
            gradient: None
        }
    }

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
        let mut transposed = self.clone();
        let new_shape = [transposed.shape[1], transposed.shape[0], transposed.shape[2], transposed.shape[3]];
        transposed.shape = new_shape;
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                let target = flatten_indices(&transposed, [j, i, 0, 0]);
                transposed.values.swap(flatten_indices(&self, [i, j, 0,  0]), target);
            }
        }
        transposed
    }

    pub fn is_bigger(&self, scalar: f32) -> Tensor {
        let mut t = Tensor::zeros(self.shape);
        for i in 0..t.values.len() {
            if t.values[i] > scalar {
                t.values[i] = 1.;
            }
        }
        t
    }

    pub fn is_smaller(&self, scalar: f32) -> Tensor {
        let mut t = Tensor::zeros(self.shape);
        for i in 0..t.values.len() {
            if t.values[i] < scalar {
                t.values[i] = 1.;
            }
        }
        t
    }

    pub fn sum(&self, axis: usize) -> Tensor {

        let mut n_shape = self.shape;
        n_shape[axis] = 1;
        let mut t = Tensor::zeros(n_shape);

        let mut sum_index = 1;
        for i in 0..axis {
            sum_index *= self.shape[i]
        }
        let next_dim_index = sum_index * self.shape[axis];

        let mut curr_pos = 0;
        for i in 0..self.values.len() {

            t.values[curr_pos] += self.values[i];

            let next_ind = (curr_pos+1) % sum_index != 0;
            let new_axis = (i+1) % next_dim_index == 0;
            if next_ind || new_axis {
                curr_pos += 1;
            } else {
                curr_pos -= sum_index-1 ;
            }
        }
        t
    }

    pub fn norm(&mut self) -> &mut Tensor {
        let mean = self.values.iter().sum::<f32>()/(self.values.len() as f32);
        let std: f32 = (self.values.iter().map(|val| (val - mean).powi(2)).sum::<f32>()/(self.values.len() as f32)).sqrt();
        for i in 0..self.values.len() {
            self.values[i] = (self.values[i] - mean) / std;
        }
        self
    }
}

fn length_flat_indices(indices: [usize; 4]) -> usize {
    let mut total_length = 1;
    for i in 0..indices.len() {
        total_length *= indices[i];
    }
    total_length
}

fn flatten_indices(tensor: &Tensor, indices: [usize; 4]) -> usize {

    assert!(indices.iter().zip(tensor.shape.iter()).all(|(a,b)| a < b));

    let mut index = 0;
    let mut mul = 1;
    
    for i in 0..tensor.shape.len() {
        index += indices[i]*mul;
        mul *= tensor.shape[i];
    }
    index
}

impl Index<[usize; 4]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 4]) -> &f32 {
        &self.values[flatten_indices(&self, indices)]
    }
}

impl Index<[usize; 3]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 3]) -> &f32 {
        &self.values[flatten_indices(&self, [indices[0], indices[1], indices[2], 0])]
    }
}

impl Index<[usize; 2]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 2]) -> &f32 {
        &self.values[flatten_indices(&self, [indices[0], indices[1], 0, 0])]
    }
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.values[flatten_indices(&self, [index, 0, 0, 0])]
    }
}

impl IndexMut<[usize; 4]> for Tensor {

    fn index_mut(&mut self, indices: [usize; 4]) -> &mut f32 {
        let index = flatten_indices(&self, indices);
        &mut self.values[index]
    }
}

impl IndexMut<[usize; 3]> for Tensor {

    fn index_mut(&mut self, indices: [usize; 3]) -> &mut f32 {
        let index = flatten_indices(&self, [indices[0], indices[1], indices[2], 0]);
        &mut self.values[index]
    }
}

impl IndexMut<[usize; 2]> for Tensor {

    fn index_mut(&mut self, indices: [usize; 2]) -> &mut f32 {
        let index = flatten_indices(&self, [indices[0], indices[1], 0, 0]);
        &mut self.values[index]
    }
}

impl IndexMut<usize> for Tensor {

    fn index_mut(&mut self, index: usize) -> &mut f32 {
        let index = flatten_indices(&self, [index, 0, 0, 0]);
        &mut self.values[index]
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        //TODO: Allow broadcasting
        assert!(self.shape[0]==rhs.shape[1]);
        assert!(self.shape[2]==rhs.shape[2]);
        assert!(self.shape[3]==rhs.shape[3] || self.shape[3] == 1);

        let mut t = Tensor::zeros([rhs.shape[0], self.shape[1], self.shape[2], rhs.shape[3]]);
        for b in 0..rhs.shape[3] {
            let mut bs = b;
            if self.shape[3] == 1 {
                bs = 0;
            }
            for c in 0..self.shape[2] {
                // MATMUL
                for i in 0..rhs.shape[0] {
                    for j in 0..self.shape[1] {
                        let mut val = 0.;
                        for k in 0..self.shape[0] { 
                            val += self[[k, j, c, bs]]*rhs[[i, k, c, b]];
                        }
                        t[[i, j, c, b]] = val;
                    }
                }
            }
        }
        t
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {

        let mut new_vals = self.values.clone();
        for i in 0..new_vals.len() {
            new_vals[i] = new_vals[i]*rhs;
        }
        Tensor::new(new_vals, self.shape)
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        let mut new_vals = rhs.values.clone();
        for i in 0..new_vals.len() {
            new_vals[i] = new_vals[i]*self;
        }
        Tensor::new(new_vals, rhs.shape)
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, mut rhs: Tensor) -> Tensor {
        for i in 0..rhs.values.len() {
            rhs.values[i] = rhs.values[i]*self;
        }
        rhs
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(mut self, rhs: f32) -> Tensor {

        for i in 0..self.values.len() {
            self.values[i] /= rhs;
        }
        self
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Tensor {

        let mut t = self.clone();
        for i in 0..t.values.len() {
            t.values[i] /= rhs;
        }
        t
    }
}
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && self.shape[2] == rhs.shape[2]);

        if self.shape[3] == rhs.shape[3] {
            let mut t = self.clone();
            for i in 0..t.values.len() {
                t.values[i] += rhs.values[i];
            }
            t
        } else if self.shape[3] == 1 {
            let mut t = rhs.clone();
            for i in 0..t.values.len() {
                t.values[i] += self.values[i % self.values.len()];
            }
            t
        } else if rhs.shape[3] == 1 {
            let mut t = self.clone();
            for i in 0..t.values.len() {
                t.values[i] += rhs.values[i % rhs.values.len()];
            }
            t
        } else {
            panic!();
        }
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape.iter().zip(rhs.shape.iter()).all(|(a,b)| a == b));

        let mut t = self.clone();
        for i in 0..t.values.len() {
            t.values[i] -= rhs.values[i];
        }
        t
    }
}

impl Sub<&Tensor> for &mut Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape.iter().zip(rhs.shape.iter()).all(|(a,b)| a == b));

        let mut t = self.clone();
        for i in 0..t.values.len() {
            t.values[i] -= rhs.values[i];
        }
        t
    }
}

impl SubAssign<f32> for &mut Tensor {
    fn sub_assign(&mut self, rhs: f32) {

        for i in 0..self.values.len() {
            self.values[i] -= rhs;
        }
    }
}

impl SubAssign<&Tensor> for &mut Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        assert!(self.shape.iter().zip(rhs.shape.iter()).all(|(a,b)| a == b));

        for i in 0..self.values.len() {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        assert!(self.shape.iter().zip(rhs.shape.iter()).all(|(a,b)| a == b));

        for i in 0..self.values.len() {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl Display for Tensor {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "Tensor with shape: {:?}\n", self.shape)?;
        for j in 0..self.shape[1] {
            for i in 0..self.shape[0] {
                write!(f, "{} ", self[[i, j, 0, 0]])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    } 
}