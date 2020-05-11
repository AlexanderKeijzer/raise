extern crate raise;

use raise::tensor::Tensor;
use std::time::Instant;

fn main() {


    let e = Tensor::new(vec![0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], [1, 10, 1, 1]);
    let eh = Tensor::new(vec![-0.1208, -0.0378,  0.0611,  0.1810,  0.2944, -0.0533, -0.0004, -0.3365,
        0.0637,  0.3152], [1, 10, 1, 1]);

    let test = &e*&(&eh - &eh.logsumexp(1));
    println!("{}", eh.logsumexp(1));
    println!("{}", test.transpose());


    let t = Tensor::uniform_int([1, 1, 1, 1000], 0, 10);
    let t2 = t.to_one_hot(0);
    println!("{}", t2);
    /*
    {
        let size: i64 = 4*50*28*28*42000;
        println!("{}", size);
        let t = Tensor::rand([1, 50, 1, 42000]);
        let t2 = Tensor::rand([28*28, 1, 1, 42000]);
        let start = Instant::now();
        let t3 = (&t*&t2).sum(3);
        println!("{}ms", start.elapsed().as_millis());
        let start = Instant::now();
        let t4 = t.outermean3(&t2);
        println!("{}ms", start.elapsed().as_millis());
        assert_eq!(t3, t4);
    }\*/
    {
        let t = Tensor::rand([28*28, 50, 1, 1]);
        let t2 = Tensor::rand([1, 28*28, 1, 42000]);
        let start = Instant::now();
        let t3 = &t*&t2;
        println!("{:?}", t3.shape);
        println!("{}ms", start.elapsed().as_millis());
    }

}