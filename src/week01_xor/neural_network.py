"""Small neural-network building blocks for the week-one XOR exercise."""
import numpy as np
def relu(z):
    return np.maximum(0, z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def initialize_parameters(seed=42):
    rng=np.random.default_rng(seed)
    parameters = {
        "W1":rng.standard_normal((2,4))*0.1,
        "b1": np.zeros((1,4)),
        "W2":rng.standard_normal((4,1))*0.1,
        "b2":np.zeros((1,1))

    }
    return parameters
def forward(x,parameters):
    Z1=x@parameters["W1"]+parameters["b1"]
    A1=relu(Z1)
    Z2=A1@parameters["W2"]+parameters["b2"]
    A2=sigmoid(Z2)
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }
    return A2, cache
