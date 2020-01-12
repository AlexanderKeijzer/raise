mod tensor;
mod layers;
mod activiations;
mod ops;
mod losses;

use layers::layer::Layer;
use layers::linear::Linear;
use activiations::relu::ReLU;
use tensor::Tensor;
use losses::loss::Loss;
use losses::mse::MSE;


fn main() {

    // Init data
    let input = Tensor::rand(&[4, 1]);
    let target = Tensor::ones(&[10, 1]);

    // Init network
    let mut l1 = Linear::new([20, 4]);
    let mut r = ReLU::new(20);
    let mut l2 = Linear::new([10, 20]);
    let mut mse = MSE::new(10);

    // Forward pass network
    let a = l1.fwd(input);
    let b = r.fwd(a);
    let c = l2.fwd(b);
    let loss = mse.fwd(c, &target);

    // Print Loss
    print!("{}", loss);

    // Backward pass network
    mse.bwd(&target);
    l2.bwd(mse.get_input());
    r.bwd(l2.get_input());
    l1.bwd(r.get_input());

    //Optimizer
}

