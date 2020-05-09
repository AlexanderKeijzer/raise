//! # raise
//!
//! TODO

//#![feature(proc_macro_hygiene)]
pub mod tensor;
pub mod layers;
pub mod activiations;
pub mod ops;
pub mod losses;
pub mod optimizers;
pub mod data;

use tensor::Tensor;
use std::time::Instant;
use layers::layer::Layer;
use losses::loss::Loss;
use optimizers::optimizer::Optimizer;
use data::dataloader::DataLoader;

extern crate rand;

fn accuracy(prediction: &Tensor, target: &Tensor) -> f32 {
    prediction.argmax(1).equals_(&target.argmax(1)).mean_all()
}

pub fn fit(epochs: usize, model: &mut dyn Layer, loss: &mut dyn Loss, optimizer: &mut dyn Optimizer, train_loader: &DataLoader, valid_loader: &DataLoader) {
    for epoch in 0..epochs {
        let start = Instant::now();
        let mut accu = 0.;
        for (x, y) in train_loader.batches() {

            let y_hat = model.fwd(x);
            accu += accuracy(&y_hat, &y);

            let _curr_loss = loss.fwd(y_hat, &y);

            let loss_grad = loss.bwd(y);
            model.bwd(loss_grad);

            optimizer.step(model.get_parameters());
        }

        let mut accu_val = 0.;
        for (x, y) in valid_loader.batches() {
            let y_hat = model.fwd(x);
            accu_val += accuracy(&y_hat, &y);
        }

        accu /= train_loader.len() as f32;
        accu_val /= valid_loader.len() as f32;

        println!("Epoch {}: Train Accuracy: {:.3}, Valid Accuracy: {:.3}, Elapsed Time: {:.2}s", epoch, accu, accu_val, start.elapsed().as_secs_f32());
    }
}