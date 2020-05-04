extern crate rand;
extern crate rand_distr;

use std::ops::{Index, IndexMut, Mul, Add, Sub, Div, SubAssign};
use std::fmt;
use std::fmt::Display;
use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Debug, Clone, PartialEq)]
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

    pub fn count(shape: [usize; 4]) -> Tensor {
        Tensor {
            shape: shape,
            values: (0..length_flat_indices(shape)).map(|a| a as f32).collect(),
            gradient: None
        }
    }

    pub fn pow(mut self, exponent: f32) -> Tensor {
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
    pub fn transpose(&self) -> Tensor {
        let mut transposed = self.clone();
        let new_shape = [transposed.shape[1], transposed.shape[0], transposed.shape[2], transposed.shape[3]];
        transposed.shape = new_shape;
        for b in 0..self.shape[3] {
            for c in 0..self.shape[2] {
                for j in 0..self.shape[1] {
                    for i in 0..self.shape[0] {
                        let target = flatten_indices(&transposed, [j, i, c, b]);
                        transposed.values.swap(flatten_indices(&self, [i, j, c, b]), target);
                    }
                }
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

    pub fn reduce_axis<F>(&self, axis: usize, init: f32, reduction: F) -> Tensor 
        where F: Fn(f32, f32) -> f32 {

        let mut n_shape = self.shape;
        n_shape[axis] = 1;
        let mut t = Tensor::new(vec![init; length_flat_indices(n_shape)], n_shape);

        let mut sum_index = 1;
        for i in 0..axis {
            sum_index *= self.shape[i]
        }
        let next_dim_index = sum_index * self.shape[axis];

        let mut curr_pos = 0;
        for i in 0..self.values.len() {

            t.values[curr_pos] = reduction(t.values[curr_pos], self.values[i]);

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

    pub fn sum(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, 0., |a, b| a + b)
    }

    pub fn max(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, f32::NEG_INFINITY, |a, b| if a > b {a} else {b})
    }

    pub fn min(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, f32::INFINITY, |a, b| if a < b {a} else {b})
    }

    pub fn norm(mut self) -> Tensor {
        let mean = self.values.iter().sum::<f32>()/(self.values.len() as f32);
        let std: f32 = (self.values.iter().map(|val| (val - mean).powi(2)).sum::<f32>()/(self.values.len() as f32)).sqrt();
        for i in 0..self.values.len() {
            self.values[i] = (self.values[i] - mean) / std;
        }
        self
    }

    pub fn exp(mut self) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].exp();
        }
        self
    }

    pub fn ln(mut self) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].ln();
        }
        self
    }

    pub fn logsumexp(&self) -> Tensor {
        let max = self.max(1);
        (&max-self).exp().sum(1).ln() + max
    }

    pub fn item_size(&self) -> usize {
        self.shape[0]*self.shape[1]*self.shape[2]
    }

    pub fn get_minibatch(&self, position: usize, amount: usize) -> Tensor {
        let last_item = position*self.item_size() + amount*self.item_size();
        assert!(last_item <= self.values.len());
        let shape = [self.shape[0], self.shape[1], self.shape[2], amount];
        Tensor::new(self.values[position*self.item_size()..last_item].to_vec(), shape)
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

/*
Indexing
*/

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

/*
Multiplication
*/

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

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(mut self, rhs: f32) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i]*rhs;
        }
        self
    }
}

/*
Division
*/

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

/*
Addition
*/

enum TensorType<'a> {
    Owned(Tensor),
    Reference(&'a Tensor)
}

impl<'a> TensorType<'a> {

    pub fn is_owned(&self) -> bool {
        matches!(*self, TensorType::Owned(_))
    }

    pub fn is_ref(&self) -> bool {
        !self.is_owned()
    }

    pub fn get_shape(&self) -> [usize; 4] {
        match self {
            TensorType::Owned(t) => t.shape,
            TensorType::Reference(t) => t.shape,
        }
    }

    pub fn get_owned(self) -> Tensor {
        match self {
            TensorType::Owned(t) => t,
            TensorType::Reference(t) => panic!("Requested owned Tensor but TensorType is not owned!"),
        }
    }

    pub fn get_reference(&'a self) -> &'a Tensor {
        match self {
            TensorType::Owned(t) => t,
            TensorType::Reference(t) => *t,
        }
    }
}

/*
Broadcasting addition takes 11x the time of full addition. We can do better than this... Needs rethink.
*/
fn broadcast<T>(t1: TensorType, t2: TensorType, operation: T) -> Tensor
    where T: Fn(f32, f32) -> f32 {
    let s_t1 = t1.get_shape();
    let s_t2 = t2.get_shape();

    // Not broadcasting shortcut
    if s_t1.iter().zip(s_t2.iter()).all(|(a,b)| a == b) {
        let mut t_owner;
        let t_reference;
        if t1.is_owned() {
            t_owner = t1.get_owned();
            t_reference = t2.get_reference();
        } else if t2.is_owned() {
            t_owner = t2.get_owned();
            t_reference = t1.get_reference();
        } else {
            t_owner = t1.get_reference().clone();
            t_reference = t2.get_reference();
        }
        for i in 0..t_owner.values.len() {
            t_owner.values[i] = operation(t_owner.values[i], t_reference.values[i]);
        }
        return t_owner;
    }

    let mut broadcaster = 0;
    let mut broadcast_axis = [false, false, false, false];

    for i in 0..s_t1.len() {
        if s_t1[i] > s_t2[i] {
            assert!(s_t2[i] == 1, "Axis {} is not of equal length or 1!", i);
            if broadcaster == 2 {
                panic!("Can only broadcast one way: One of the Tensors in this operation needs to be bigger or as big as the other Tensor in every dimension!");
            }
            broadcast_axis[i] = true;
            broadcaster = 1;
        } else if s_t1[i] < s_t2[i] {
            assert!(s_t1[i] == 1, "Axis {} is not of equal length or 1!", i);
            if broadcaster == 1 {
                panic!("Can only broadcast one way: One of the Tensors in this operation needs to be bigger or as big as the other Tensor in every dimension!");
            }
            broadcast_axis[i] = true;
            broadcaster = 2;
        }
    }

    let mut t_owner;
    let t_broadcaster;
    if broadcaster == 1 {
        if t1.is_owned() {
            t_owner = t1.get_owned();
        } else {
            t_owner = t1.get_reference().clone();
        }
        t_broadcaster = t2.get_reference();
    } else if broadcaster == 2 {
        if t2.is_owned() {
            t_owner = t2.get_owned();
        } else {
            t_owner = t2.get_reference().clone();
        }
        t_broadcaster = t1.get_reference();
    } else {
        if t1.is_owned() {
            t_owner = t1.get_owned();
            t_broadcaster = t2.get_reference();
        } else if t2.is_owned() {
            t_owner = t2.get_owned();
            t_broadcaster = t1.get_reference();
        } else {
            t_owner = t1.get_reference().clone();
            t_broadcaster = t2.get_reference();
        }
    }

    // The code below is very inefficient because it's not optimized well by the compiler (which is no surprise)
    // It might be better (but worse memory-wise) to create a larger vector instead
    
    let mut not_bc_size = [1, 1, 1, 1];
    for i in 1..not_bc_size.len() {
        if !broadcast_axis[i-1] {
            not_bc_size[i] = not_bc_size[i-1]*t_owner.shape[i-1];
        } else {
            not_bc_size[i] = not_bc_size[i-1];
        }
    }
    let mut step_size = [1, 1, 1, 1];
    for i in 1..t_owner.shape.len() {
        step_size[i] = step_size[i-1]*t_owner.shape[i-1];
    }

    let mut current = 0;
    for i in 0..t_owner.values.len() {
        t_owner.values[i] = operation(t_owner.values[i], t_broadcaster.values[current]); //i % t_broadcaster.values.len()

        for j in (0..step_size.len()).rev() {
            if (i+1) % step_size[j] == 0 {
                current += 1;
                if broadcast_axis[j] {
                    current -= not_bc_size[j];
                }
                break;
            }
        }
    }
    t_owner
}

pub fn add(t1: &Tensor, t2: &Tensor) -> Tensor {
    broadcast(TensorType::Reference(t1), TensorType::Reference(t2), |a, b| a + b)
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

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(mut self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && self.shape[2] == rhs.shape[2]);

        if self.shape[3] == rhs.shape[3] {
            for i in 0..self.values.len() {
                self.values[i] += rhs.values[i];
            }
            self
        } else if self.shape[3] == 1 {
            let mut t = rhs.clone();
            for i in 0..t.values.len() {
                t.values[i] += self.values[i % self.values.len()];
            }
            t
        } else if rhs.shape[3] == 1 {
            for i in 0..self.values.len() {
                self.values[i] += rhs.values[i % rhs.values.len()];
            }
            self
        } else {
            panic!();
        }
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(mut self, mut rhs: Tensor) -> Tensor {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && self.shape[2] == rhs.shape[2]);

        if self.shape[3] == rhs.shape[3] {
            for i in 0..self.values.len() {
                self.values[i] += rhs.values[i];
            }
            self
        } else if self.shape[3] == 1 {
            for i in 0..rhs.values.len() {
                rhs.values[i] += self.values[i % self.values.len()];
            }
            rhs
        } else if rhs.shape[3] == 1 {
            for i in 0..self.values.len() {
                self.values[i] += rhs.values[i % rhs.values.len()];
            }
            self
        } else {
            panic!();
        }
    }
}

/*
Subtraction
*/

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && self.shape[2] == rhs.shape[2]);

        if self.shape[3] == rhs.shape[3] {
            let mut t = self.clone();
            for i in 0..t.values.len() {
                t.values[i] -= rhs.values[i];
            }
            t
        } else if self.shape[3] == 1 {
            let mut t = rhs.clone();
            for i in 0..t.values.len() {
                t.values[i] -= self.values[i % self.values.len()];
            }
            t
        } else if rhs.shape[3] == 1 {
            let mut t = self.clone();
            for i in 0..t.values.len() {
                t.values[i] -= rhs.values[i % rhs.values.len()];
            }
            t
        } else {
            panic!();
        }
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(mut self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && self.shape[2] == rhs.shape[2]);

        if self.shape[3] == rhs.shape[3] {
            for i in 0..self.values.len() {
                self.values[i] -= rhs.values[i];
            }
            self
        } else if self.shape[3] == 1 {
            let mut t = rhs.clone();
            for i in 0..t.values.len() {
                t.values[i] -= self.values[i % self.values.len()];
            }
            t
        } else if rhs.shape[3] == 1 {
            for i in 0..self.values.len() {
                self.values[i] -= rhs.values[i % rhs.values.len()];
            }
            self
        } else {
            panic!();
        }
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, mut rhs: Tensor) -> Tensor {
        assert!(self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && self.shape[2] == rhs.shape[2]);

        if self.shape[3] == rhs.shape[3] {
            for i in 0..rhs.values.len() {
                rhs.values[i] -= self.values[i];
            }
            rhs
        } else if self.shape[3] == 1 {
            for i in 0..rhs.values.len() {
                rhs.values[i] -= self.values[i % self.values.len()];
            }
            rhs
        } else if rhs.shape[3] == 1 {
            let mut t = self.clone();
            for i in 0..t.values.len() {
                t.values[i] -= rhs.values[i % rhs.values.len()];
            }
            t
        } else {
            panic!();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation() {
        assert_eq!(Tensor::ones([3, 3, 1, 1]), Tensor::ones([3, 3, 1, 1]))
    }
    #[test]
    fn indexing() {
        let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], [3, 2, 1, 2]);
        assert_eq!(t[[0, 0, 0, 0]], 0.);
        assert_eq!(t[[1, 0, 0, 0]], 1.);
        assert_eq!(t[[1, 1, 0, 1]], 10.);

        assert_eq!(t[[1, 1, 0]], 4.);
        assert_eq!(t[[1, 1]], 4.);
        assert_eq!(t[1], 1.);
    }

    #[test]
    fn addition() {
        use std::time::Instant;

        let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], [2, 3, 1, 2]);

        let t2t = Tensor::new(vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22.], [2, 3, 1, 2]);
        assert_eq!(&t+&t, t2t);

        //Broadcasting
        let tbc = Tensor::ones([2, 3, 1, 1]);
        let tbct = Tensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [2, 3, 1, 2]);
        assert_eq!(&t+&tbc, tbct);

        /*
        let testje = Tensor::count([3, 2, 1, 2]);
        let testje1 = Tensor::count([1, 2, 1, 1]);
        let testje2 = Tensor::count([1, 1, 1, 2]);
        let testje3 = Tensor::count([1, 2, 1, 2]);
        let testje4 = Tensor::count([3, 1, 1, 2]);

        println!("{}", &testje);
        //println!("{}", &testje1);
        println!("{}", add(&testje, &testje1));
        println!("{}", add(&testje, &testje2));
        println!("{}", add(&testje, &testje3));
        println!("{}", add(&testje, &testje4));
        */

        let testje = Tensor::rand([2000, 20, 1, 2]);
        let testje2 = Tensor::rand([2000, 20, 1, 1]);
        let testje_res = 2.*Tensor::ones([2000, 20, 1, 1000]);

        let start = Instant::now();
        let s2 = add(&testje,&testje2);
        println!("Broadcast: {}", start.elapsed().as_micros());
        println!("Broadcast: {}", s2.values[5]);

        let start = Instant::now();
        let s1 = &testje+&testje2;
        println!("Normal: {}", start.elapsed().as_micros());
        println!("Normal: {}", s1.values[5]);

        //println!("{}", testje);
        //println!("{}", testje2);
        //println!("{}", s1);
        //println!("{}", s2);
        //assert_eq!(s1, s2);

        //assert_eq!(add(&testje, &testje2), testje_res);
    }

    #[test]
    fn subtraction() {
        let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], [2, 3, 1, 2]);

        let t2t = Tensor::zeros([2, 3, 1, 2]);
        assert_eq!(&t-&t, t2t);

        //Broadcasting
        let tbc = Tensor::ones([2, 3, 1, 1]);
        let tbct = Tensor::new(vec![-1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [2, 3, 1, 2]);
        assert_eq!(&t-&tbc, tbct);
    }

    #[test]
    fn multiplication() {
        let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], [2, 3, 1, 2]);

        let t2t = Tensor::zeros([2, 3, 1, 2]);
        assert_eq!(&t-&t, t2t);

        //Broadcasting
        let tbc = Tensor::ones([2, 3, 1, 1]);
        let tbct = Tensor::new(vec![-1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [2, 3, 1, 2]);
        assert_eq!(&t-&tbc, tbct);
    }
}