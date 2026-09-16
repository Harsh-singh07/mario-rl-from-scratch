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
def binary_cross_entropy(predictions,targets):
    
    epsilon=1e-15
    predictions=np.clip(predictions,epsilon,1-epsilon)
    loss=-np.mean(targets*np.log(predictions)+(1-targets)*np.log(1-predictions))
    return loss
def backward(x,y,parameters,cache):
    m=x.shape[0]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    dZ2 = A2 - y
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ parameters["W2"].T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = (x.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }

    return gradients
def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]

    return parameters


