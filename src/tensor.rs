extern crate rand;
extern crate rand_distr;

use std::ops::{Index, IndexMut, Mul, Add, Sub, Div, SubAssign, Neg};
use std::fmt;
use std::fmt::Display;
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use std::cmp;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: [usize; 4],
    values: Vec<f32>,
    pub gradient: Option<Box<Tensor>>
}

// x y c s
impl Tensor {
    pub fn new(values:Vec<f32>, shape: [usize; 4]) -> Tensor {
        assert!(values.len() == length_flat_indices(shape));
        Tensor {
            shape: shape,
            values: values,
            gradient: None
        }
    }

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

    pub fn uniform(shape: [usize; 4], low: f32, high: f32) -> Tensor {
        let mut rng = thread_rng();
        let uni = Uniform::new(low, high);
        Tensor {
            shape: shape,
            values: (0..length_flat_indices(shape)).map(|_| rng.sample(uni)).collect(),
            gradient: None
        }
    }

    pub fn uniform_int(shape: [usize; 4], low: i32, high: i32) -> Tensor {
        let mut rng = thread_rng();
        let uni = Uniform::new(low, high);
        Tensor {
            shape: shape,
            values: (0..length_flat_indices(shape)).map(|_| rng.sample(uni) as f32).collect(),
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

    pub fn powf(mut self, exponent: f32) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].powf(exponent);
        }
        self
    }

    pub fn powi(mut self, exponent: i32) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].powi(exponent);
        }
        self
    }

    pub fn sum_all(&self) -> f32 {
        self.values.iter().sum::<f32>()
    }

    pub fn mean_all(&self) -> f32 {
        self.sum_all()/(self.values.len() as f32)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        //Make sure tensor is destroyed and move values instead of cloning?
        self.values.clone()
    }

    pub fn transpose_(mut self) -> Tensor {
        let new_shape = [self.shape[1], self.shape[0], self.shape[2], self.shape[3]];

        if self.shape[0] != 1 && self.shape[1] != 1 {
            let step_size = get_axis_steps(self.shape);

            for b in 0..self.shape[3] {
                for c in 0..self.shape[2] {
                    let start = b*step_size[3] + c*step_size[2];
                    for i in 0..(step_size[2]) {
                        self.values.swap(start+i, start+((i*step_size[1])%step_size[2]));
                    }
                }
            }
        }
        self.shape = new_shape;
        self
    }

    pub fn transpose(&self) -> Tensor {
        let transposed = self.clone();
        transposed.transpose_()
    }

    pub fn is_between_(mut self, min: f32, max: f32) -> Tensor {
        for i in 0..self.values.len() {
            if self.values[i] > min && self.values[i] < max {
                self.values[i] = 1.;
            } else {
                self.values[i] = 0.;
            }
        }
        self
    }

    pub fn is_between(&self, scalar: f32) -> Tensor {
        let t = self.clone();
        t.is_bigger_(scalar)
    }

    pub fn is_bigger_(mut self, scalar: f32) -> Tensor {
        for i in 0..self.values.len() {
            if self.values[i] > scalar {
                self.values[i] = 1.;
            } else {
                self.values[i] = 0.;
            }
        }
        self
    }

    pub fn is_bigger(&self, scalar: f32) -> Tensor {
        let t = self.clone();
        t.is_bigger_(scalar)
    }

    pub fn is_smaller_(mut self, scalar: f32) -> Tensor {
        for i in 0..self.values.len() {
            if self.values[i] < scalar {
                self.values[i] = 1.;
            } else {
                self.values[i] = 0.;
            }
        }
        self
    }

    pub fn is_smaller(&self, scalar: f32) -> Tensor {
        let t = self.clone();
        t.is_smaller_(scalar)
    }

    pub fn equals_(self, tensor: &Tensor) -> Tensor {
        broadcast(TensorType::Owned(self), TensorType::Reference(tensor), |a, b| {
            (a == b) as i32 as f32
        })
    }

    pub fn equals(&self, tensor: &Tensor) -> Tensor {
        let t= self.clone();
        t.equals_(tensor)
    }

    pub fn reduce_axis<F>(&self, axis: usize, init: f32, reduction: F) -> Tensor 
        where F: Fn(f32, f32) -> f32 {

        let mut n_shape = self.shape;
        n_shape[axis] = 1;
        let mut t = Tensor::new(vec![init; length_flat_indices(n_shape)], n_shape);

        let step_size = get_axis_steps(self.shape);
        let step_size_red = get_axis_steps(t.shape);

        // Similar to broadcasting, could merge
        for b in 0..self.shape[3] {
            let bb = if axis == 3 {0} else {b*step_size_red[3]};
            for c in 0..self.shape[2] {
                let cb = if axis == 2 {bb} else {bb + c*step_size_red[2]};
                for j in 0..self.shape[1] {
                    let jb = if axis == 1 {cb} else {cb + j*step_size_red[1]};
                    for i in 0..self.shape[0] {
                        let ib = if axis == 0 {jb} else {jb + i};
                        let index = b*step_size[3]+c*step_size[2]+j*step_size[1]+i;
                        t.values[ib] = reduction(t.values[ib], self.values[index]);
                    }
                }
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

    pub fn mean(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, 0., |a, b| a + b/(self.shape[axis] as f32))
    }

    pub fn std(&self) -> f32 {
        let mean = self.mean_all();
        (self.values.iter().map(|val| (val - mean).powi(2)).sum::<f32>()/(self.values.len() as f32)).sqrt()
    }

    pub fn norm(&mut self) -> (f32, f32) {
        let mean = self.mean_all();
        let std: f32 = (self.values.iter().map(|val| (val - mean).powi(2)).sum::<f32>()/(self.values.len() as f32)).sqrt();
        self.norm_with(mean, std);
        (mean, std)
    }

    pub fn norm_with(&mut self, mean: f32, std: f32) {
        for i in 0..self.values.len() {
            self.values[i] = (self.values[i] - mean) / std;
        }
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

    pub fn logsumexp(&self, axis: usize) -> Tensor {
        let max = self.max(axis);
        (self-&max).exp().sum(axis).ln() + max
    }

    pub fn argmax(&self, axis: usize) -> Tensor {
        // COPY FROM REDUCE AXIS, TRY TO MERGE
        let mut n_shape = self.shape;
        n_shape[axis] = 1;
        let mut result = Tensor::new(vec![-1.; length_flat_indices(n_shape)], n_shape);
        let mut max_val = Tensor::new(vec![f32::NEG_INFINITY; length_flat_indices(n_shape)], n_shape);

        let step_size = get_axis_steps(self.shape);
        let step_size_red = get_axis_steps(result.shape);

        // Similar to broadcasting, could merge
        for b in 0..self.shape[3] {
            let bb = if axis == 3 {0} else {b*step_size_red[3]};
            for c in 0..self.shape[2] {
                let cb = if axis == 2 {bb} else {bb + c*step_size_red[2]};
                for j in 0..self.shape[1] {
                    let jb = if axis == 1 {cb} else {cb + j*step_size_red[1]};
                    for i in 0..self.shape[0] {
                        let ib = if axis == 0 {jb} else {jb + i};
                        let index = b*step_size[3]+c*step_size[2]+j*step_size[1]+i;
                        if self.values[index] > max_val.values[ib] {
                            max_val.values[ib] = self.values[index];
                            result.values[ib] = match axis {
                                0 => i as f32,
                                1 => j as f32,
                                2 => c as f32,
                                3 => b as f32,
                                _ => panic!()
                            };
                        }
                    }
                }
            }
        }
        result
    }

    pub fn clamp(mut self, min: f32, max: f32) -> Tensor {
        for i in 0..self.values.len() {
            if self[i] < min {
                self.values[i] = min;
            } else if self[i] > max {
                self.values[i] = max;
            } 
        }
        self
    }

    pub fn clamp_min(mut self, min: f32) -> Tensor {
        for i in 0..self.values.len() {
            if self[i] < min {
                self.values[i] = min;
            }
        }
        self
    }

    pub fn clamp_max(mut self, max: f32) -> Tensor {
        for i in 0..self.values.len() {
            if self[i] > max {
                self.values[i] = max;
            } 
        }
        self
    }

    pub fn outersum3(&self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[0] == 1);
        //Accept even if not a column vector as outer product is obv row*column
        assert!(rhs.shape[0] == 1 || rhs.shape[1] == 1);
        assert!(self.shape[2] == 1 && self.shape[2] == 1); //TODO: handle channel
        let s_max = cmp::max(self.shape[3], rhs.shape[3]);

        let mut result = Tensor::zeros([rhs.shape[0]*rhs.shape[1], self.shape[1], 1, 1]);

        for s in 0..s_max {
            let s_s = &self.values[cmp::min(s, self.shape[3]-1)*self.shape[1]..(cmp::min(s, self.shape[3]-1)+1)*self.shape[1]];
            let r_s = &rhs.values[cmp::min(s, rhs.shape[3]-1)*rhs.shape[0]*rhs.shape[1]..(cmp::min(s, rhs.shape[3]-1)+1)*rhs.shape[0]*rhs.shape[1]];
            for j in 0..result.shape[1] {
                let row_pos = j*result.shape[0];
                let res = &mut result.values[row_pos..row_pos+result.shape[0]];
                for i in 0..result.shape[0] {
                    res[i] += r_s[i]*s_s[j];
                }
            }
        }
        result
    }

    pub fn outermean3(&self, rhs: &Tensor) -> Tensor {
        self.outersum3(rhs)/(cmp::max(self.shape[3], rhs.shape[3]) as f32)
    }

    pub fn to_one_hot(&self, axis: usize) -> Tensor {
        assert!(self.shape[axis] == 1);
        let max = self.values.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap().round() as usize;
        let mut n_shape = self.shape;
        n_shape[axis] = max+1;
        let mut result = Tensor::zeros(n_shape);
        for s in 0..self.shape[3] {
            for c in 0..self.shape[2] {
                for j in 0..self.shape[1] {
                    for i in 0..self.shape[0] {
                        let n_axis = self[[i, j, c, s]].round() as usize;
                        let s_r = if axis == 3 {n_axis} else {s};
                        let c_r = if axis == 2 {n_axis} else {c};
                        let j_r = if axis == 1 {n_axis} else {j};
                        let i_r = if axis == 0 {n_axis} else {i};
                        result[[i_r, j_r, c_r, s_r]] = 1.;
                    }
                }
            }
        }
        result
    }

    pub fn item_size(&self) -> usize {
        self.shape[0]*self.shape[1]*self.shape[2]
    }

    pub fn get_minibatch(&self, start: usize, end: usize) -> Tensor {
        assert!(end*self.item_size() <= self.values.len());
        let shape = [self.shape[0], self.shape[1], self.shape[2], end-start];
        Tensor::new(self.values[start*self.item_size()..end*self.item_size()].to_vec(), shape)
    }

    pub fn split(mut self, size_first: f32) -> (Tensor, Tensor) {
        let len_first = (size_first*(self.shape[3] as f32)).floor() as usize;
        let second = self.values.split_off(len_first*self.item_size());
        (Tensor::new(self.values, [self.shape[0], self.shape[1], self.shape[2], len_first]),
         Tensor::new(second, [self.shape[0], self.shape[1], self.shape[2], self.shape[3] - len_first]))
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

    assert!(indices.iter().zip(tensor.shape.iter()).all(|(a,b)| a < b), "Trying to index at {:?} in a tensor of size {:?}", indices, tensor.shape);

    let mut index = 0;
    let mut mul = 1;
    
    for i in 0..tensor.shape.len() {
        index += indices[i]*mul;
        mul *= tensor.shape[i];
    }
    index
}

fn get_axis_steps(shape: [usize; 4]) -> [usize; 4]{
    let mut step_size = [1, 1, 1, 1];
    for i in 1..shape.len() {
        step_size[i] = step_size[i-1]*shape[i-1];
    }
    step_size
}

fn for_each_sample<T>(t1: &Tensor, t2: &Tensor, result: &mut Tensor, operation: T)
where T: Fn(&[f32], &[f32], &mut [f32]) {
    let as_t1 =  get_axis_steps(t1.shape);
    let as_t2 =  get_axis_steps(t2.shape);
    let as_result =  get_axis_steps(result.shape);
    for b in 0..t2.shape[3] {
        let b_t1 = if t1.shape[3] == 1 {0} else {b};
        let b_t2 = if t2.shape[3] == 1 {0} else {b};
        for c in 0..t1.shape[2] {
            let s_t1 = b_t1*as_t1[3]+c*as_t1[2];
            let s_t2 = b_t2*as_t2[3]+c*as_t2[2];
            let s_result = b*as_result[3]+c*as_result[2];

            let sample_t1 = &t1.values[s_t1..s_t1+as_t1[2]];
            let sample_t2 = &t2.values[s_t2..s_t2+as_t2[2]];
            let out = &mut result.values[s_result..s_result+as_result[2]];
            operation(sample_t1, sample_t2, out);
        }
    }
}

/*
Broadcasting addition is now faster than equal size addition!
*/
fn broadcast<T>(t1: TensorType, t2: TensorType, operation: T) -> Tensor
    where T: Fn(f32, f32) -> f32 {
    let s_t1 = t1.get_shape();
    let s_t2 = t2.get_shape();

    // Shortcutting equal size tensors is not siginificantly faster

    let mut broadcasted = 0;
    let mut broadcast_axis = [false, false, false, false];

    for i in 0..s_t1.len() {
        if s_t1[i] > s_t2[i] {
            assert!(s_t2[i] == 1, "Axis {} is not of equal length or 1!", i);
            if broadcasted == 2 {
                panic!("Can only broadcast one way: One of the Tensors in this operation needs to be bigger or as big as the other Tensor in every dimension!");
            }
            broadcast_axis[i] = true;
            broadcasted = 1;
        } else if s_t1[i] < s_t2[i] {
            assert!(s_t1[i] == 1, "Axis {} is not of equal length or 1!", i);
            if broadcasted == 1 {
                panic!("Can only broadcast one way: One of the Tensors in this operation needs to be bigger or as big as the other Tensor in every dimension!");
            }
            broadcast_axis[i] = true;
            broadcasted = 2;
        }
    }

    // Shrink or seperate this

    let mut t_owner;
    let t_broadcaster;
    if broadcasted == 1 {
        if t1.is_owned() {
            t_owner = t1.get_owned();
        } else {
            t_owner = t1.get_reference().clone();
        }
        t_broadcaster = t2.get_reference();
    } else if broadcasted == 2 {
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


    let step_size = get_axis_steps(t_owner.shape);
    let step_size_bc = get_axis_steps(t_broadcaster.shape);

    // Might be possible to compact this up with a smart iterator
    for b in 0..t_owner.shape[3] {
        let bb = if broadcast_axis[3] {0} else {b*step_size_bc[3]};
        for c in 0..t_owner.shape[2] {
            let cb = if broadcast_axis[2] {bb} else {bb + c*step_size_bc[2]};
            for j in 0..t_owner.shape[1] {
                let jb = if broadcast_axis[1] {cb} else {cb + j*step_size_bc[1]};
                for i in 0..t_owner.shape[0] {
                    let ib = if broadcast_axis[0] {jb} else {jb + i};
                    let index = b*step_size[3]+c*step_size[2]+j*step_size[1]+i;
                    t_owner.values[index] = operation(t_owner.values[index], t_broadcaster.values[ib]);
                }
            }
        }
    }
    t_owner
}

/*
Multiplication
*/

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape[2]==rhs.shape[2]); // Allow broadcasting?
        assert!(self.shape[3]==rhs.shape[3] || self.shape[3] == 1 || rhs.shape[3] == 1);

        if self.shape[0] == rhs.shape[0] && self.shape[1] == rhs.shape[1] && (self.shape[0] == 1 || self.shape[1] == 1) {
            element_wise_mul(self, rhs)
        } else if rhs.shape[0] == 1 {
            matrix_vector_mul(self, rhs)
        } else if self.shape[0] == 1 && rhs.shape[1] == 1 {
            vector_vector_mul(self, rhs)
        } else {
            assert!(self.shape[0]==rhs.shape[1], "Invalid multiplication!");
            mat_mul(self, rhs)
        }
    }
}

fn mat_mul(m1: &Tensor, m2: &Tensor) -> Tensor {
    let mut result = Tensor::zeros([m2.shape[0], m1.shape[1], m1.shape[2], cmp::max(m1.shape[3], m2.shape[3])]);

    for_each_sample(m1, m2, &mut result, |mat1, mat2, out| {
        for i in 0..m2.shape[0] {
            for j in 0..m1.shape[1] {
                let mut res: f32 = 0.0;
                for k in 0..m1.shape[0] {
                    // res = mat[k,j] * mat[i,k]
                    res += mat1[k+j*m1.shape[0]]*mat2[i+k*m2.shape[0]];
                }
                // mat[i,j]
                out[i+j*m2.shape[0]] = res;
            }
        }
    });
    result
}

fn matrix_vector_mul(matrix: &Tensor, vector: &Tensor) -> Tensor {
    let mut result = Tensor::zeros([vector.shape[0], matrix.shape[1], matrix.shape[2], cmp::max(matrix.shape[3], vector.shape[3])]);

    for_each_sample(matrix, vector, &mut result, |mat, vec, out| {
        for j in 0..matrix.shape[1] {
            let row: &[f32] = &mat[j*matrix.shape[0]..((j+1)*matrix.shape[0])];
            
            /*
            let mut res: f32 = 0.0;
            for i in 0..matrix.shape[0] {
                // res = mat[i,j] * vec[i]
                res += row[i]*vec[i];
            }
            // vec[j] = res
            out[j] = res;
            */
            
            
            out[j] = row.iter().zip(vec.iter()).map(|(r, v)| r*v).sum::<f32>();
        }
    });
    result
}

fn vector_vector_mul(v1: &Tensor, v2: &Tensor) -> Tensor {
    let mut result = Tensor::zeros([v2.shape[0], v1.shape[1], v1.shape[2], cmp::max(v1.shape[3], v2.shape[3])]);
    for_each_sample(v1, v2, &mut result, |vec1, vec2, out| {
        for j in 0..v1.shape[1] {
            for i in 0..v2.shape[0] {
                // mat[i,j] = vec[i] * vec[j]
                out[i+j*v2.shape[0]] = vec1[j]*vec2[i];
            }
        }
    });
    result
}

fn element_wise_mul(v1: &Tensor, v2: &Tensor) -> Tensor {
    let mut result = v1.clone();
    for_each_sample(v1, v2, &mut result, |_, vec2, out| {
        for i in 0..vec2.len() {
            out[i] *= vec2[i];
        }
    });
    result
}

// We might be able to do things inplace with sqaure matrices and vec-vec multiplication
impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        (&self).mul(rhs)
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        self.mul(&rhs)
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        (&self).mul(&rhs)
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

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape.iter().sum::<usize>() >= rhs.shape.iter().sum());
        broadcast(TensorType::Reference(self), TensorType::Reference(rhs), |a, b| a / b)
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {
        assert!(self.shape.iter().sum::<usize>() >= rhs.shape.iter().sum());
        broadcast(TensorType::Owned(self), TensorType::Reference(rhs), |a, b| a / b)
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Tensor {
        assert!(self.shape.iter().sum::<usize>() >= rhs.shape.iter().sum());
        broadcast(TensorType::Reference(self), TensorType::Owned(rhs), |a, b| a / b)
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Tensor {
        assert!(self.shape.iter().sum::<usize>() >= rhs.shape.iter().sum());
        broadcast(TensorType::Owned(self), TensorType::Owned(rhs), |a, b| a / b)
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

impl Div<Tensor> for f32 {
    type Output = Tensor;

    fn div(self, mut rhs: Tensor) -> Tensor {

        for i in 0..rhs.values.len() {
            rhs.values[i] = self/rhs[i];
        }
        rhs
    }
}

impl Div<&Tensor> for f32 {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {

        let mut t = rhs.clone();
        for i in 0..t.values.len() {
            t.values[i] = self/t[i];
        }
        t
    }
}

/*
Addition
*/

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        broadcast(TensorType::Reference(self), TensorType::Reference(rhs), |a, b| a + b)
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        broadcast(TensorType::Owned(self), TensorType::Reference(rhs), |a, b| a + b)
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        broadcast(TensorType::Reference(self), TensorType::Owned(rhs), |a, b| a + b)
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        broadcast(TensorType::Owned(self), TensorType::Owned(rhs), |a, b| a + b)
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(mut self, rhs: f32) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] += rhs;
        }
        self
    }
}

impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, mut rhs: Tensor) -> Tensor {
        for i in 0..rhs.values.len() {
            rhs.values[i] += self;
        }
        rhs
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Tensor {
        let mut t = self.clone();
        for i in 0..t.values.len() {
            t.values[i] += rhs;
        }
        t
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        let mut t = rhs.clone();
        for i in 0..t.values.len() {
            t.values[i] += self;
        }
        t
    }
}

/*
Subtraction
*/

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        broadcast(TensorType::Reference(self), TensorType::Reference(rhs), |a, b| a - b)
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        broadcast(TensorType::Owned(self), TensorType::Reference(rhs), |a, b| a - b)
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        broadcast(TensorType::Reference(self), TensorType::Owned(rhs), |a, b| a - b)
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        broadcast(TensorType::Owned(self), TensorType::Owned(rhs), |a, b| a - b)
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(mut self, rhs: f32) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] -= rhs;
        }
        self
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

impl SubAssign<Tensor> for &mut Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
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

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(mut self) -> Tensor {
        for i in 0..self.values.len() {
            self.values[i] = -self.values[i];
        }
        self
    }
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

impl Index<[usize; 1]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 1]) -> &f32 {
        &self.values[flatten_indices(&self, [indices[0], 0, 0, 0])]
    }
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.values[index]
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

impl IndexMut<[usize; 1]> for Tensor {

    fn index_mut(&mut self, indices: [usize; 1]) -> &mut f32 {
        let index = flatten_indices(&self, [indices[0], 0, 0, 0]);
        &mut self.values[index]
    }
}

impl IndexMut<usize> for Tensor {

    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.values[index]
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

enum TensorType<'a> {
    Owned(Tensor),
    Reference(&'a Tensor)
}

impl<'a> TensorType<'a> {

    pub fn is_owned(&self) -> bool {
        matches!(*self, TensorType::Owned(_))
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
            TensorType::Reference(_) => panic!("Requested owned Tensor but TensorType is not owned!"),
        }
    }

    pub fn get_reference(&'a self) -> &'a Tensor {
        match self {
            TensorType::Owned(t) => t,
            TensorType::Reference(t) => *t,
        }
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
        let t = Tensor::count([3, 2, 1, 2]);
        assert_eq!(t[[0, 0, 0, 0]], 0.);
        assert_eq!(t[[1, 0, 0, 0]], 1.);
        assert_eq!(t[[1, 1, 0, 1]], 10.);

        assert_eq!(t[[1, 1, 0]], 4.);
        assert_eq!(t[[1, 1]], 4.);
        assert_eq!(t[1], 1.);
    }

    #[test]
    fn addition() {

        let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], [2, 3, 1, 2]);

        let t2t = Tensor::new(vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22.], [2, 3, 1, 2]);
        assert_eq!(&t+t.clone(), t2t);

        //Broadcasting
        let tbc = Tensor::ones([2, 3, 1, 1]);
        let tbct = Tensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [2, 3, 1, 2]);
        assert_eq!(t+tbc, tbct);
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
        //TODO
    }

    #[test]
    fn comparison() {
        //TODO
    }
}