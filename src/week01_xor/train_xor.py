"""Entry point for training the week-one XOR model."""


# if __name__ == "__main__":
#     raise SystemExit("XOR training implementation is ready for week 01.")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
training_steps = 10_000
loss_history = []

for step in range(training_steps):
    predictions, cache = forward(x, parameters)

    loss = binary_cross_entropy(predictions, y)
    loss_history.append(loss)

    gradients = backward(x, y, parameters, cache)

    parameters = update_parameters(
        parameters,
        gradients,
        learning_rate,
    )

    if step % 500 == 0:
        print(f"Step {step:5d} | Loss: {loss:.4f}")

predictions, cache = forward(x, parameters)
final_loss = binary_cross_entropy(predictions, y)

binary_predictions = (predictions >= 0.5).astype(int)

print(f"\nFinal loss: {final_loss:.4f}")
print("\nFinal XOR predictions:")

for input_values, probability, predicted_class, expected in zip(
    x,
    predictions,
    binary_predictions,
    y,
):
    print(
        f"{input_values} -> "
        f"probability={probability[0]:.4f}, "
        f"predicted={predicted_class[0]}, "
        f"expected={expected[0]}"
    )
output_dir = Path("artifacts/week01")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Binary cross-entropy loss")
plt.title("XOR Neural Network Training Loss")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()

plot_path = output_dir / "xor_loss.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")

print(f"\nLoss graph saved to: {plot_path}")