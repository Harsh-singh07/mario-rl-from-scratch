# Mario RL From Scratch

Experiments for learning reinforcement learning from first principles.

## Week 01: XOR

The first exercise builds and trains a small neural network on the XOR truth table.

Run it with:

```powershell
python src/week01_xor/train_xor.py
```

Generated outputs belong in `artifacts/week01/`.
# Mario RL From Scratch

I am building a reinforcement-learning agent for Super Mario Bros
without using ML frameworks such as PyTorch or TensorFlow.

## Week 1 — Neural Network From Scratch

### Goal

Build a small neural network using only NumPy and train it to solve XOR.

### Network Design

```text
2 input neurons → 4 hidden ReLU neurons → 1 sigmoid output neuron
What I Implemented
- Forward propagation
- ReLU and sigmoid activation functions
- Binary cross-entropy loss
- Backpropagation
- Gradient descent
- Training-loss visualization
Result
Final loss: ADD_YOUR_FINAL_LOSS_HERE
Input	Expected	Prediction
[0, 0]	0	0.0
[0, 1]	1	1.0
[1, 0]	1	1.0
[1, 1]	0	0.0


Training Loss
for output refer\artifacts\week01\xor_loss.png

What I Learned
A neural network learns by calculating prediction error, sending that
error backward through the layers, and updating weights to reduce loss.
This is the same core idea that I will later use in a Deep Q-Network
for the Mario agent.