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
use losses::loss::Loss;
use losses::mse::MSE;
use optimizers::optimizer::Optimizer;
use optimizers::sgd::SGD;
use data::loader;


fn main() {

    //let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], [3, 2, 1, 2]);
    //let ts = t.sum(3);
    //print!("{}",ts);
    //let t2 = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8.], [3, 3, 1, 1]);
    //let t3 = &t*&t2;
    //print!("{}",&t*2.);
    //print!("{}",t2);
    //print!("{}",t3);

    
    let (mut input, target) = loader::read("C:/Users/alty/Downloads/MNIST/train.csv");
    println!("loaded");
    print!("{}",input.T());
    input.norm();
    print!("{}",input.T());
    print!("{}",input.to_vec().iter().sum::<f32>());
    print!("{}",target.T());

    // Init data
    //let input = Tensor::rand([1, 4, 1, 1]);
    //let target = Tensor::rand([1, 10, 1, 1]);

    // Init network
    let mut l1 = Linear::new([input.shape[1], 20]);
    let mut r = ReLU::new(20);
    let mut l2 = Linear::new([20, target.shape[1]]);
    let mut mse = MSE::new(target.shape[1]);

    let opt = SGD::new(0.0005);

    let start = Instant::now();

    for _ in 0..200 {
        // Forward pass network
        let a = l1.fwd(input.clone());
        let b = r.fwd(a);
        let c = l2.fwd(b);
        let loss = mse.fwd(c, &target);

        // Print Loss
        print!("{}\n", loss);

        // Backward pass network
        mse.bwd(&target);
        l2.bwd(mse.get_input());
        r.bwd(l2.get_input());
        l1.bwd(r.get_input());

        //Optimizer
        opt.step(l1.get_parameters());
        opt.step(l2.get_parameters());
    }
    print!("{}", start.elapsed().as_micros());

    let test_val = &input.to_vec()[..784];
    let test = Tensor::new(test_val.clone().to_vec(), [1, 784, 1, 1]);

    let test_targ_val = &target.to_vec()[..10];
    let test_targ = Tensor::new(test_targ_val.clone().to_vec(), [1, 10, 1, 1]);

    let a = l1.fwd(test.clone());
    let b = r.fwd(a);
    let c = l2.fwd(b);

    print!("{}",test_targ.T());
    print!("{}",c.T());
}

