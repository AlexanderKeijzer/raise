# RAISE

RAISE is a neural network library writting in Rust from scratch.
RAISE includes a seperate experimental autograd package called RAISE Graph which can be found [here](https://github.com/AlexanderKeijzer/raise-graph).

## Installation

RAISE needs to be installed with RAISE Graph on the same directory level.

```TOML
[dependencies]
rand="0.7.3"
rand_distr = "0.2.2"
dyn-clone = "1.0.1"
inline-python = "0.5.1"
raise-graph = { path = "../raise-graph" }
```

## Usage

For an example how to use RAISE please see the examples folder.
For usable performance always build and run with the `--release` flag.

### MNIST 

[mnist.rs](https://github.com/AlexanderKeijzer/raise/blob/master/examples/mnist.rs) downloads the mnist dataset and trains a simple neural network.
The accuracy resulting accuracy should be around 96%.

Initializing a model works similarly  to PyTorch and TensorFlow:

```Rust
let hidden_layer = 50;

let mut model = Sequential::new(vec![
    Box::new(Linear::new([train_set.input_shape()[1], hidden_layer], "relu")),
    Box::new(ReLU::new()),
    Box::new(Linear::new([hidden_layer, train_set.target_shape()[1]], "")),
]);

let mut loss_func = CrossEntropy::new();
let mut optimizer = SGD::new(0.05);

let batch_size = 64;
```

Training a model in RAISE only needs a couple of lines. No need to manually write a training loop. However, since everything is written in Rust you can easily take a look at how things work under the hood!

```Rust
let (mean, std) = train_set.norm_input();
valid_set.norm_input_with(mean, std);

let train_loader = DataLoader::new(train_set, batch_size, true);
let valid_loader = DataLoader::new(valid_set, batch_size, false);

fit(5, &mut model, &mut loss_func, &mut optimizer, &train_loader, &valid_loader);
```

This shoud result in an output similar this:
```bash
Epoch 0: Train Accuracy: 0.852, Train Loss: 0.513, Valid Accuracy: 0.932, Valid Loss: 0.244, Elapsed Time: 4.80s
Epoch 1: Train Accuracy: 0.925, Train Loss: 0.260, Valid Accuracy: 0.944, Valid Loss: 0.202, Elapsed Time: 4.67s
Epoch 2: Train Accuracy: 0.942, Train Loss: 0.201, Valid Accuracy: 0.935, Valid Loss: 0.225, Elapsed Time: 4.73s
Epoch 3: Train Accuracy: 0.952, Train Loss: 0.170, Valid Accuracy: 0.955, Valid Loss: 0.162, Elapsed Time: 4.69s
Epoch 4: Train Accuracy: 0.956, Train Loss: 0.152, Valid Accuracy: 0.959, Valid Loss: 0.150, Elapsed Time: 4.68s
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.