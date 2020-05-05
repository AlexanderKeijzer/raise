#![feature(proc_macro_hygiene)]
mod tensor;
mod layers;
mod activiations;
mod ops;
mod losses;
mod optimizers;
mod data;
mod tensor_view;

use tensor::Tensor;
use std::time::Instant;
use layers::layer::Layer;
use layers::linear::Linear;
use activiations::relu::ReLU;
use losses::loss::Loss;
use losses::mse::MSE;
use losses::cross_entropy::CrossEntropy;
use optimizers::optimizer::Optimizer;
use optimizers::sgd::SGD;
use data::loader;
use data::plot;

extern crate rand;

use rand::Rng;


fn main() {

    //let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8.], [1, 3, 1, 3]);
    //let ts = t.logsumexp();
    //print!("{}",ts);
    //let t2 = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8.], [3, 3, 1, 1]);
    //let t3 = &t*&t2;
    //print!("{}",&t*2.);
    //print!("{}",t2);
    //print!("{}",t3);
    
    let (mut input, target) = loader::read("C:/Users/alty/Downloads/MNIST/train.csv");
    println!("Loaded dataset");
    input = input.norm();

    // Init data
    let hidden_layer = 50;

    // Init network
    let mut l1 = Linear::new([input.shape[1], hidden_layer]);
    let mut r = ReLU::new(hidden_layer);
    let mut l2 = Linear::new([hidden_layer, target.shape[1]]);
    let mut mse = MSE::new(target.shape[1]);

    let opt = SGD::new(0.01);

    let start = Instant::now();

    let bs = 32;//input.shape[3];

    let mut loss_list = Vec::new();

    for epoch in 0..1 {
        for mbi in 0..input.shape[3]/bs {

            let x = input.get_minibatch(mbi, bs);
            let y = target.get_minibatch(mbi, bs);

            // Forward pass network
            //let start = Instant::now();
            let a = l1.fwd(x);
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            let b = r.fwd(a);
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            let c = l2.fwd(b);
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            let loss = mse.fwd(c, &y);
            //println!("{}", start.elapsed().as_micros());
            loss_list.push(loss);

            // Print Loss
            println!("Epoch: {}, Minibatch: {}, Loss: {}", epoch, mbi, loss);

            // Backward pass network
            // TODO: cleanup inputs during bwd pass? bwd should return the ownership of its input and set its field to None
            //let start = Instant::now();
            mse.bwd(&y);
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            l2.bwd(mse.get_input());
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            r.bwd(l2.get_input());
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            l1.bwd(r.get_input());
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();

            // Optimizer
            opt.step(l1.get_parameters());
            //println!("{}", start.elapsed().as_micros());
            //let start = Instant::now();
            opt.step(l2.get_parameters());
            //println!("{}", start.elapsed().as_micros());
            // Zero gradients?
        }
    }
    println!("{}", start.elapsed().as_micros());
    /*
    plot::plot(loss_list);

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
        println!("Result: {}", found.logsumexp());
        let e = found.clone().exp();
        println!("Result: {}", (&e/(e.sum(1))).ln());
        println!("Result: {}", (&found - found.logsumexp()).exp());
        plot::imshow(&inp, Some([28, 28]));
    }
    */
}

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

