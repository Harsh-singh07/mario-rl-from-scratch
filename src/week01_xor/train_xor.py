"""Entry point for training the week-one XOR model."""


# if __name__ == "__main__":
#     raise SystemExit("XOR training implementation is ready for week 01.")
import numpy as np
from neural_network import initialize_parameters,forward , binary_cross_entropy,backward,update_parameters
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

learning_rate = 0.1

predictions, cache = forward(x, parameters)
initial_loss = binary_cross_entropy(predictions, y)

print(f"Initial loss: {initial_loss:.4f}")

for step in range(500):
    predictions, cache = forward(x, parameters)

    loss = binary_cross_entropy(predictions, y)
    gradients = backward(x, y, parameters, cache)

    parameters = update_parameters(
        parameters,
        gradients,
        learning_rate,
    )

predictions, cache = forward(x, parameters)
final_loss = binary_cross_entropy(predictions, y)

print(f"Final loss:   {final_loss:.4f}")

print("\nPredictions after 500 updates:")
for input_values, prediction in zip(x, predictions):
    print(f"{input_values} -> {prediction[0]:.4f}")
