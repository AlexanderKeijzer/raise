mod tensor;
mod layers;
mod activiations;
mod ops;
mod losses;
mod optimizers;
mod data;

use std::time::Instant;
use layers::layer::Layer;
use layers::linear::Linear;
use activiations::relu::ReLU;
use tensor::Tensor;
use losses::loss::Loss;
use losses::mse::MSE;
use optimizers::optimizer::Optimizer;
use optimizers::sgd::SGD;
use data::loader;


fn main() {

    let t = Tensor::new(vec![0., 1., 2., 3., 4., 5., 6., 7., 8.], &[3, 3]);
    print!("{}",t)
    /*
    let (data, target) = loader::read("C:/Users/alty/Downloads/MNIST/train.csv");

    // Init data
    //let input = Tensor::rand(&[4, 1]);
    //let target = Tensor::rand(&[10, 1]);

    // Init network
    let mut l1 = Linear::new([20, 4]);
    let mut r = ReLU::new(20);
    let mut l2 = Linear::new([10, 20]);
    let mut mse = MSE::new(10);

    let opt = SGD::new(0.1);

    let start = Instant::now();

    for _ in 0..100 {
        // Forward pass network
        let a = l1.fwd(data.clone());
        let b = r.fwd(a);
        let c = l2.fwd(b);
        let loss = mse.fwd(c, &target);

        // Print Loss
        //print!("{}\n", loss);

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
    */
}

