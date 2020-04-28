# RAISE Design Documentation

## Tensor representation
Tensors representation is variable depending on their us in RAISE. Tensors are always at least 4D, however, in most cases some of these layers will be hidden or not used.

Each dimension representents the following feature of a Tensor:
- Input data 1D: [- ,Sample, Channel, Data]
- Input data 2D: [Sample, Channel, Y-Data, X-Data]
- Affine layer weights: [-, Feature, Y-Weights, X-Weights]
- Affine layer biases: [-, -, Feature, Biases]

Tensors store data in a 1D vector. Data will be stored in opposite order as shown above e.g.:

[X1Y1C1S, X2Y1C1S1, X1Y2C1S1, X2Y2C1S1, X1Y1C2S1 ...]

## Tensor multiplication
