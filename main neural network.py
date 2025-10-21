import os, json, numpy as np, pickle , renumpying
from renumpying import sqroot, sum_func, exp , unifromity , lcg , uniformity , zero
from random_number_generators import lcg , uniformity , randint
import redoing_numpy_arrays as rna
# --- Load or initialize move history ---
if os.path.exists("moves.json") and os.path.getsize("moves.json") > 0:
    with open("moves.json", "r") as f:
        move_data = json.load(f)
else:
    move_data = []

history_length = len(move_data[-1]["history"]) + 1 if move_data else 5
input_size = history_length * 3

# --- Network architecture ---
hidden1_size = 64
hidden2_size = 32
hidden3_size = 16
output_size = 3

# --- Activation functions ---
def relu(x):
    return max(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)
def softmax(x):
    exp_x = exp(x - max(x))
    return exp_x / sum_func(exp_x)

# --- Xavier initialization ---
def xavier_init(size_in, size_out):
    limit = sqroot(6 / (size_in + size_out))
    return uniformity(-limit, limit, size=(size_in, size_out), seed_start=1)

  

# --- Initialize weights and biases ---
W1 = xavier_init(input_size, hidden1_size)
b1 = zero(hidden1_size)
W2 = xavier_init(hidden1_size, hidden2_size)
b2 = zero(hidden2_size)
W3 = xavier_init(hidden2_size, hidden3_size)
b3 = zero(hidden3_size)
W4 = xavier_init(hidden3_size, output_size)
b4 = zero(output_size)

# --- Generate random RPS input ---
def generate_random_rps_input(k):
    moves = randint(0, 3, k)
    x = []
    for move in moves:
        if move == 0: x.extend([1,0,0])
        elif move == 1: x.extend([0,1,0])
        else: x.extend([0,0,1])
    return np.array(x), moves.tolist()

# --- Generate random dataset ---
num_samples = 500
X = []
Y = np.zeros((num_samples, 3))
for i in range(num_samples):
    x_vec, moves_hist = generate_random_rps_input(history_length)
    X.append(x_vec)
    move = np.random.randint(0,3)
    Y[i, move] = 1
X = np.array(X)

# --- Cross-entropy loss ---
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# --- Training loop ---
epochs = 2000
learning_rate = 0.05

for epoch in range(epochs):
    total_loss = 0
    for i in range(num_samples):
        x = X[i]
        y = Y[i]

        # --- Forward pass ---
        z1 = np.dot(x, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)
        z3 = np.dot(a2, W3) + b3
        a3 = relu(z3)
        z4 = np.dot(a3, W4) + b4
        y_pred = softmax(z4)

        # --- Compute loss ---
        total_loss += cross_entropy(y, y_pred)

        # --- Backpropagation ---
        dz4 = y_pred - y
        dW4 = np.outer(a3, dz4)
        db4 = dz4

        dz3 = (W4 @ dz4) * relu_derivative(z3)
        dW3 = np.outer(a2, dz3)
        db3 = dz3

        dz2 = (W3 @ dz3) * relu_derivative(z2)
        dW2 = np.outer(a1, dz2)
        db2 = dz2

        dz1 = (W2 @ dz2) * relu_derivative(z1)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        # --- Gradient updates ---
        W4 -= learning_rate * dW4
        b4 -= learning_rate * db4
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/num_samples:.4f}")

# --- Test prediction ---
x_test_vec, x_test_history = generate_random_rps_input(history_length)
z1 = np.dot(x_test_vec, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = relu(z2)
z3 = np.dot(a2, W3) + b3
a3 = relu(z3)
z4 = np.dot(a3, W4) + b4
y_pred = softmax(z4)

predicted_move = int(np.argmax(y_pred))
your_move = (predicted_move + 1) % 3

# --- Save weights and biases ---
all_my_weights_biases = {
    "layer_1": {"weights": W1.tolist(), "biases": b1.tolist()},
    "layer_2": {"weights": W2.tolist(), "biases": b2.tolist()},
    "layer_3": {"weights": W3.tolist(), "biases": b3.tolist()},
    "layer_4": {"weights": W4.tolist(), "biases": b4.tolist()},
}

with open("weights_biases.pkl", "wb") as f:
    pickle.dump(all_my_weights_biases, f)

# --- Print predictions ---
print("\nPredicted opponent move:", ["Rock","Paper","Scissors"][predicted_move])
print("Your move to win:", ["Rock","Paper","Scissors"][your_move])

# --- Save test entry ---
move_entry = {
    "history": x_test_history,
    "predicted_move": int(predicted_move),
    "your_move": int(your_move)
}
move_data.append(move_entry)
with open("moves.json", "w") as f:
    json.dump(move_data, f, indent=2)
