extern crate raise;

use std::fs::File;
use raise::fit;
use raise::tensor::Tensor;
use raise::layers::linear::Linear;
use raise::activiations::relu::ReLU;
use raise::losses::cross_entropy::CrossEntropy;
use raise::optimizers::sgd::SGD;
use raise::data::dataset::DataSet;
use raise::data::dataloader::DataLoader;
use raise::layers::sequential::Sequential;

fn main() {
    let mut dataset = read("data/train.csv");
    println!("Loaded dataset");

    // Init data
    let hidden_layer = 50;

    // Init model
    let mut model = Sequential::new(vec![
        Box::new(Linear::new([dataset.input_shape()[1], hidden_layer], "pytorch")),
        Box::new(ReLU::new()),
        Box::new(Linear::new([hidden_layer, dataset.target_shape()[1]], "pytorch")),
    ]);

    let mut loss_func = CrossEntropy::new();
    let mut optimizer = SGD::new(0.005);

    let batch_size = 64;

    // Init data loaders
    dataset.norm_input();
    let (train_set, valid_set) = dataset.split(0.8);
    let train_loader = DataLoader::new(train_set, batch_size, true);
    let valid_loader = DataLoader::new(valid_set, batch_size, false);

    fit(50, &mut model, &mut loss_func, &mut optimizer, &train_loader, &valid_loader);
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