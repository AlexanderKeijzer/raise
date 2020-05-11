extern crate raise;

use raise::tensor::Tensor;
use raise::layers::linear::Linear;
use raise::activiations::relu::ReLU;
use raise::losses::loss::Loss;
use raise::losses::cross_entropy::CrossEntropy;
use raise::layers::layer::Layer;
use raise::layers::sequential::Sequential;

fn main() {


    // Init data
    let input_layer = 28*28;
    let hidden_layer = 50;
    let output_layer = 10;

    // Init model
    let mut model = Sequential::new(vec![
        Box::new(Linear::new([input_layer, hidden_layer], "pytorch")),
        Box::new(ReLU::new()),
        //Box::new(Print::new("ReLU")),
        Box::new(Linear::new([hidden_layer, output_layer], "pytorch")),
    ]);
    //let mut l = Linear::new([input_layer, hidden_layer], "pytorch");
    let mut loss_func = CrossEntropy::new();

    let y;
    {
        let tmp = Tensor::uniform_int([1, 1, 1, 1000], 0, 10);
        y = tmp.to_one_hot(1);
    }
    let t = Tensor::rand([1, input_layer, 1, 1000]);

    //let t1 = l.fwd(t);
    //println!("Input:   Mean {:.4}, STD {:.4}", t1.mean_all(), t1.std());

    println!("Input:     Mean {:.4}, STD {:.4}", t.mean_all(), t.std());
    let tout = model.fwd(t);
    println!("Output:    Mean {:.4}, STD {:.4}", tout.mean_all(), tout.std());
    //println!("{}", tout.transpose());
    //println!("{}", y.transpose());
    let loss = loss_func.fwd(tout, &y);
    println!("Loss:      {}", loss);

    let t_loss = loss_func.bwd(y);
    println!("Loss grad: Mean {:.4}, STD {:.4}", t_loss.mean_all(), t_loss.std());

    let t_bwd = model.bwd(t_loss);
    println!("In grad:   Mean {:.4}, STD {:.4}", t_bwd.mean_all(), t_bwd.std());
    let params = model[0].as_mut().get_parameters();
    let g = params[0].gradient.as_ref().unwrap().as_ref();
    println!("W grad:    Mean {:.4}, STD {:.4}", g.mean_all(), g.std());

    let g = params[1].gradient.as_ref().unwrap().as_ref();
    println!("B grad:    Mean {:.4}, STD {:.4}", g.mean_all(), g.std());
}