#![feature(proc_macro_hygiene)]

extern crate raise;

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
use inline_python::{python, Context};

fn main() {
    let (mut train_set, mut valid_set) = read_pickle_py("data/mnist.pkl");
    println!("Loaded dataset");

    // Init data
    let hidden_layer = 50;

    // Init model
    let mut model = Sequential::new(vec![
        Box::new(Linear::new([train_set.input_shape()[1], hidden_layer], "relu")),
        Box::new(ReLU::new()),
        Box::new(Linear::new([hidden_layer, train_set.target_shape()[1]], "")),
    ]);

    let mut loss_func = CrossEntropy::new();
    let mut optimizer = SGD::new(0.05);

    let batch_size = 64;

    // Init data loaders
    let (mean, std) = train_set.norm_input();
    valid_set.norm_input_with(mean, std);
    
    let train_loader = DataLoader::new(train_set, batch_size, true);
    let valid_loader = DataLoader::new(valid_set, batch_size, false);

    lr_find(&mut model, &mut loss_func, &mut optimizer, &train_loader);

    fit(5, &mut model, &mut loss_func, &mut optimizer, &train_loader, &valid_loader);
}

fn read_pickle_py(file_path: &str) -> (DataSet, DataSet) {
    
    let py_context: Context = python! {
        import pickle

        file = open('file_path, "rb")
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(file, encoding="latin-1")
        file.close()
        
        x_train = x_train.flatten().tolist()
        y_train = y_train.flatten().tolist()
        x_valid = x_valid.flatten().tolist()
        y_valid = y_valid.flatten().tolist()
    };

    let x_train = py_context.get::<Vec<f32>>("x_train");
    let y_train = py_context.get::<Vec<f32>>("y_train");
    let x_valid = py_context.get::<Vec<f32>>("x_valid");
    let y_valid = py_context.get::<Vec<f32>>("y_valid");
    (DataSet::new(Tensor::new(x_train, [1, 28*28, 1, 50000]), Tensor::new(y_train, [1, 1, 1, 50000]).to_one_hot(1)),
    DataSet::new(Tensor::new(x_valid, [1, 28*28, 1, 10000]), Tensor::new(y_valid, [1, 1, 1, 10000]).to_one_hot(1)))

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