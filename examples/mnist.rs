extern crate raise;
extern crate serde_pickle;

use std::fs::File;
use raise::*;
use raise::tensor::Tensor;
use raise::layers::linear::Linear;
use raise::activiations::relu::ReLU;
use raise::losses::cross_entropy::CrossEntropy;
use raise::optimizers::sgd::SGD;
use raise::data::dataset::DataSet;
use raise::data::dataloader::DataLoader;
use raise::layers::sequential::Sequential;
use serde_pickle as pkl;

fn main() {
    let (mut train_set, mut valid_set) = read_pickle("data/mnist2.pkl");
    println!("Loaded dataset");

    // Init data
    let hidden_layer = 50;

    // Init model
    let mut model = Sequential::new(vec![
        Box::new(Linear::new([train_set.input_shape()[1], hidden_layer], "pytorch")),
        Box::new(ReLU::new()),
        Box::new(Linear::new([hidden_layer, train_set.target_shape()[1]], "pytorch")),
    ]);

    let mut loss_func = CrossEntropy::new();
    let mut optimizer = SGD::new(0.04); //0.04

    let batch_size = 64;

    // Init data loaders
    train_set.norm_input();
    valid_set.norm_input(); // Should norm with train values
    //let (train_set, valid_set) = dataset.split(0.8);
    let train_loader = DataLoader::new(train_set, batch_size, true);
    let valid_loader = DataLoader::new(valid_set, batch_size, false);

    //lr_find(&mut model, &mut loss_func, &mut optimizer, &train_loader);

    fit(50, &mut model, &mut loss_func, &mut optimizer, &train_loader, &valid_loader);
}

fn read_pickle(file_path: &str) -> (DataSet, DataSet) {
    let file = File::open(file_path).unwrap();

    let mut train_val: Vec<f32> = Vec::new();
    let mut train_targ: Vec<f32> = Vec::new();
    let mut valid_val: Vec<f32> = Vec::new();
    let mut valid_targ: Vec<f32> = Vec::new();
    if let pkl::Value::Tuple(mut data) = pkl::value_from_reader(file).unwrap() {
        if let pkl::Value::Tuple(valid) = data.pop().unwrap() {
            decode_set(&mut valid_val, &mut valid_targ, valid);
        } else {
            panic!();
        }
        if let pkl::Value::Tuple(train) = data.pop().unwrap() {
            decode_set(&mut train_val, &mut train_targ, train);
        } else {
            panic!();
        }
    } else {
        panic!();
    }
    (DataSet::new(Tensor::new(train_val, [1, 28*28, 1, 50000]), Tensor::new(train_targ, [1, 1, 1, 50000]).to_one_hot(1)),
    DataSet::new(Tensor::new(valid_val, [1, 28*28, 1, 10000]), Tensor::new(valid_targ, [1, 1, 1, 10000]).to_one_hot(1)))
    
}

fn decode_set(x_list: &mut Vec<f32>, y_list: &mut Vec<f32>, mut set: Vec<pkl::Value>) {
    if let pkl::Value::List(y) = set.pop().unwrap() {
        for j in 0..y.len() {
            let v = y[j].clone();
            y_list.push(match v {
                pkl::Value::I64(f) => f as f32,
                _ => { println!("{}", v); panic!(); }
            });
        }
    } else {
        panic!();
    }
    if let pkl::Value::List(x) = set.pop().unwrap() {
        for i in 0..x.len() {
            let sublist = match x[i].clone() {
                pkl::Value::List(sublist) => sublist,
                _ => panic!()
            };
            for j in 0..sublist.len() {
                let v = sublist[j].clone();
                x_list.push(match v {
                    pkl::Value::F64(f) => f as f32,
                    _ => { println!("{}", v); panic!(); }
                });
            }
        }
    } else {
        panic!();
    }
}

fn read(file_path: &str) -> DataSet {
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut data: Vec<f32> = Vec::new();
    let mut target: Vec<f32> = Vec::new();
    let mut n_records = 0;
    for result in rdr.records() {
        let record = result.unwrap();
        let mut line: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();

        let curr_target = line.remove(0).round() as usize;
        target.append(&mut vec![0.; curr_target]);
        target.push(1.);
        target.append(&mut vec![0.; 9-curr_target]);

        data.append(&mut line);
        n_records += 1;
    }
    let data_tensor = Tensor::new(data, [1, 784, 1, n_records]);
    let target_tensor = Tensor::new(target, [1, 10, 1, n_records]);
    DataSet::new(data_tensor, target_tensor)
}