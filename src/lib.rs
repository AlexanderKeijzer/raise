//! # raise
//!
//! TODO

#![feature(proc_macro_hygiene)]
pub mod tensor;
pub mod layers;
pub mod activiations;
pub mod ops;
pub mod losses;
pub mod optimizers;
pub mod data;

extern crate rand;

use tensor::Tensor;
use std::time::Instant;
use layers::layer::Layer;
use losses::loss::Loss;
use optimizers::optimizer::Optimizer;
use data::dataloader::DataLoader;
use data::plot;

pub fn fit(epochs: usize, model: &mut dyn Layer, loss_func: &mut dyn Loss, optimizer: &mut dyn Optimizer, train_loader: &DataLoader, valid_loader: &DataLoader) {
    for epoch in 0..epochs {
        let start = Instant::now();
        
        let mut accu = 0.;
        let mut loss: f32 = 0.;
        for (x, y) in train_loader.batches() {
            let y_hat = model.fwd(x);
            accu += accuracy(&y_hat, &y);
            loss += loss_func.fwd(y_hat, &y);

            let loss_grad = loss_func.bwd(y);
            model.bwd(loss_grad);
            optimizer.step(model.get_parameters());
        }

        let mut accu_val = 0.;
        let mut loss_val = 0.;
        for (x, y) in valid_loader.batches() {
            let y_hat = model.fwd(x);
            accu_val += accuracy(&y_hat, &y);
            loss_val += loss_func.fwd(y_hat, &y);
        }

        accu /= train_loader.len() as f32;
        accu_val /= valid_loader.len() as f32;
        loss /= train_loader.len() as f32;
        loss_val /= valid_loader.len() as f32;

        println!("Epoch {}: Train Accuracy: {:.3}, Train Loss: {:.3}, Valid Accuracy: {:.3}, Valid Loss: {:.3}, Elapsed Time: {:.2}s", epoch, accu, loss, accu_val, loss_val, start.elapsed().as_secs_f32());
    }
}

pub fn lr_find(model_ref: &dyn Layer, loss_func_ref: &dyn Loss, optimizer_ref: &dyn Optimizer, train_loader: &DataLoader) {
    let mut lr_steps: Vec<f32> = (-100..0).into_iter().map(|v| 10_f32.powf((v as f32)/20.)).collect();
    let mut losses = Vec::new();
    let mut best_loss = f32::INFINITY;

    for lr in lr_steps.clone() {
        let mut model = dyn_clone::clone_box(model_ref);
        let mut loss_func = dyn_clone::clone_box(loss_func_ref);
        let mut optimizer = dyn_clone::clone_box(optimizer_ref);
        optimizer.set_learning_rate(lr);

        let mut loss = 0_f32;
        let mut steps = 0;

        for (x, y) in train_loader.batches() {
            let y_hat = model.fwd(x);
            loss = loss_func.fwd(y_hat, &y);

            if steps > 20 {
                break;
            }

            let loss_grad = loss_func.bwd(y);
            model.bwd(loss_grad);
            optimizer.step(model.get_parameters());
            steps += 1;
        }
        if loss > 4.*best_loss || !loss.is_normal() {
            lr_steps.truncate(losses.len());
            break;
        }
        if loss < best_loss {
            best_loss = loss;
        }
        losses.push(loss);
    }
    plot::plot2dlog(lr_steps, losses);
}

fn accuracy(prediction: &Tensor, target: &Tensor) -> f32 {
    prediction.argmax(1).equals_(&target.argmax(1)).mean_all()
}
