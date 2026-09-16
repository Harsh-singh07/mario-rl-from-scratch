"""Entry point for training the week-one XOR model."""


# if __name__ == "__main__":
#     raise SystemExit("XOR training implementation is ready for week 01.")
import numpy as np
from neural_network import initialize_parameters,forward
x=np.array(
    [[0,0],
     [0,1],
     [1,0],
     [1,1]],dtype=float)
y=np.array([[0],[1],[1],[0]],dtype=float)
parameters = initialize_parameters()

predictions, cache = forward(x, parameters)

print("Input shape:", x.shape)
print("Hidden-layer shape:", cache["A1"].shape)
print("Prediction shape:", predictions.shape)

print("\nPredictions before training:")
for input_values, prediction in zip(x, predictions):
    print(f"{input_values} -> {prediction[0]:.4f}")
