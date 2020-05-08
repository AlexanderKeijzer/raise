#![feature(proc_macro_hygiene)]
mod tensor;
mod layers;
mod activiations;
mod ops;
mod losses;
mod optimizers;
mod data;

use tensor::Tensor;
use std::time::Instant;
use layers::layer::Layer;
use layers::linear::Linear;
use activiations::relu::ReLU;
use activiations::mrelu::mReLU;
use losses::loss::Loss;
use losses::mse::MSE;
use losses::cross_entropy::CrossEntropy;
use optimizers::optimizer::Optimizer;
use optimizers::sgd::SGD;
use data::loader;
use data::plot;
use data::dataloader::DataLoader;
use layers::sequential::Sequential;

extern crate rand;

fn main() {
    let mut dataset = loader::read("data/train.csv");
    println!("Loaded dataset");

    // Init data
    let hidden_layer = 50;

    // Init model
    let mut model = Sequential::new(vec![
        Box::new(Linear::new([dataset.input_shape()[1], hidden_layer], "relu")),
        Box::new(ReLU::new()),
        Box::new(Linear::new([hidden_layer, dataset.target_shape()[1]], "")),
    ]);

    let mut loss_func = MSE::new();
    let mut optimizer = SGD::new(0.05);

    let batch_size = 64;

    // Init data loaders
    dataset.norm_input();
    let (train_set, valid_set) = dataset.split(0.8);
    let train_loader = DataLoader::new(train_set, batch_size, true);
    let valid_loader = DataLoader::new(valid_set, batch_size, false);

    fit(5, &mut model, &mut loss_func, &mut optimizer, &train_loader, &valid_loader);
}


fn accuracy(prediction: &Tensor, target: &Tensor) -> f32 {
    prediction.argmax(1).equals_(&target.argmax(1)).mean_all()
}

fn fit(epochs: usize, model: &mut dyn Layer, loss: &mut dyn Loss, optimizer: &mut dyn Optimizer, train_loader: &DataLoader, valid_loader: &DataLoader) {
    for epoch in 0..epochs {
        let start = Instant::now();
        let mut accu = 0.;
        for (x, y) in train_loader.batches() {

            let y_hat = model.fwd(x);
            accu += accuracy(&y_hat, &y);

            //let curr_loss = loss.fwd(y_hat, &y);

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


/*
fn to_index(tensor: &Tensor) -> usize {
    let mut res = 0;
    let mut max = f32::NEG_INFINITY;
    for i in 0..tensor.shape[1] {
        if tensor[[0, i, 0, 0]] > max {
            res = i;
            max = tensor[[0, i, 0, 0]];
        }
    }
    res
}
*/

    //let mut loss_list = Vec::new();
    //let mut accuracy_list = Vec::new();
    //let mut accuracy_valid_list = Vec::new();

    /*
    for epoch in 0..4 {
        let start = Instant::now();
        let mut accu = 0.;
        for (x, y) in train_loader.batches() {

            let y_hat = model.fwd(x);
            accu += accuracy(&y_hat, &y);
            println!("{}", accuracy(&y_hat, &y));

            let loss = ce.fwd(y_hat, &y);
            loss_list.push(loss);

            let loss_grad = ce.bwd(y);
            model.bwd(loss_grad);

            opt.step(model.get_parameters());
        }

        let mut accu_val = 0.;
        for (x, y) in valid_loader.batches() {
            let y_hat = model.fwd(x);
            accu_val += accuracy(&y_hat, &y);
        }

        accu /= train_loader.len() as f32;
        accuracy_list.push(accu);
        accu_val /= valid_loader.len() as f32;
        accuracy_valid_list.push(accu_val);

        println!("Epoch {}: Train Accuracy: {:.3}, Valid Accuracy: {:.3}, Elapsed Time: {:.2}s", epoch, accu, accu_val, start.elapsed().as_secs_f32());
    }
    */

    //plot::plot(accuracy_list);
    //plot::plot(accuracy_valid_list);
    //plot::plot(loss_list);

    /*
    let mut rng  = rand::thread_rng();
    let pos = rng.gen_range(0, input.shape[3]/5);
    let x = input.get_minibatch(pos, 5);
    let y = target.get_minibatch(pos, 5);

    let a = l1.fwd(x.clone());
    let b = r.fwd(a);
    let c = l2.fwd(b);

    for i in 0..5 {
        let inp = x.get_minibatch(i, 1);
        let targ = y.get_minibatch(i, 1);
        let found = c.get_minibatch(i, 1);
        println!("Target: {}, found: {}", to_index(&targ), to_index(&found));
        //plot::imshow(&inp, Some([28, 28]));
    }
    */