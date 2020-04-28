use std::fs::File;
use crate::tensor::Tensor;

pub fn read(file_path: &str) -> (Tensor, Tensor) {
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut data: Vec<f32> = Vec::new();
    let mut target: Vec<f32> = Vec::new();
    let mut n_records = 0;
    for result in rdr.records() {
        let record = result.unwrap();
        let mut line: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect();
        target.push(line.remove(0));
        data.append(&mut line);
        n_records += 1;
    }
    let data_tensor = Tensor::new(data, &[784, 1, 1, n_records]);
    let target_tensor = Tensor::new(target, &[1, 1, 1, n_records]);
    (data_tensor, target_tensor)
}