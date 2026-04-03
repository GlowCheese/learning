CNN strategy to train handwritting recognization on MNIST dataset.

Model = Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10) -> CEL

Code is almost similar to `mnist_mlp`, except for the `modules/model.py` implementation.
