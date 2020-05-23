#![feature(proc_macro_hygiene)]

extern crate raise;

use std::path::Path;
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

static MNIST_URL: &str = "http://deeplearning.net/data/mnist/mnist.pkl.gz";

fn main() {
    let (mut train_set, mut valid_set) = read_or_download_mnist("data/mnist.pkl.gz");
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

    fit(5, &mut model, &mut loss_func, &mut optimizer, &train_loader, &valid_loader);
}

fn read_or_download_mnist(file_path: &str) -> (DataSet, DataSet) {

    if !Path::new(file_path).exists() {
        if let Err(error) = download_mnist(file_path) {
            panic!(error.to_string());
        }
    }
    
    let py_context: Context = python! {
        import pickle
        import gzip

        with gzip.open('file_path, "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        
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

fn download_mnist(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading MNIST dataset...");
    let mut resp = reqwest::blocking::get(MNIST_URL)?;
    if resp.status().is_success() {
        let mut file = File::create(path)?;
        resp.copy_to(&mut file)?;
    } else {
        println!("{}", resp.status())
    }
    Ok(())
}