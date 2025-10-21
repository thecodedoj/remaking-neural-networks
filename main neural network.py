import os, json,pickle , renumpying
from renumpying import sqroot, sum_func, exp, zero
from random_number_generators import lcg , uniformity , randint
import redoing_numpy_arrays as rna
import remaking_the_logarithm as rtl
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
    # Convert Array to list
    if hasattr(x, 'data'):
        x = x.data
    return [max(0, xi) for xi in x]

def relu_derivative(x):
    """
    Compute the derivative of ReLU.
    Returns a native Python list of 0/1 floats.
    
    x: can be
        - a plain list of numbers
        - an instance of rna.Array
    """
    # If x is your Array class, extract the internal data
    if hasattr(x, 'data'):
        x = x.data
    
    # If x is a nested list, flatten one level (optional, depends on use)
    # Here we assume x is 1D
    return [1.0 if xi > 0 else 0.0 for xi in x]

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
    return rna.Array(x), moves

# --- Generate random dataset ---
num_samples = 500
X = []
Y = renumpying.zero((num_samples, 3))
for i in range(num_samples):
    x_vec, moves_hist = generate_random_rps_input(history_length)
    X.append(x_vec)
    move = randint(0,3)
    Y[i, move] = 1
X = rna.Array(X)

# --- Cross-entropy loss ---
def cross_entropy(y_true, y_pred):
    return -sum_func(y_true * rtl.my_log(y_pred + 1e-8))

# --- Training loop ---
epochs = 2000
learning_rate = 0.05

for epoch in range(epochs):
    total_loss = 0
    for i in range(num_samples):
        x = X[i]
        y = Y[i]

        # --- Forward pass ---
        z1 = renumpying.dot(x, W1) + b1
        a1 = relu(z1)
        z2 = renumpying.dot(a1, W2) + b2
        a2 = relu(z2)
        z3 = renumpying.dot(a2, W3) + b3
        a3 = relu(z3)
        z4 = renumpying.dot(a3, W4) + b4
        y_pred = softmax(z4)

        # --- Compute loss ---
        total_loss += cross_entropy(y, y_pred)

        # --- Backpropagation ---
        # --- Backpropagation ---

        # dz4 = y_pred - y
        dz4 = [y_pred_i - y_i for y_pred_i, y_i in zip(y_pred, y)]
        dW4 = renumpying.outer_array(a3, dz4)
        db4 = dz4

        # dz3 = dot(W4, dz4) * relu_derivative(z3)
        dz3_pre = renumpying.dot(W4, dz4)  # dot product
        dz3 = [dz3_pre[i] * relu_derivative(z3)[i] for i in range(len(dz3_pre))]
        dW3 = renumpying.outer_array(a2, dz3)
        db3 = dz3

        # dz2 = dot(W3, dz3) * relu_derivative(z2)
        dz2_pre = renumpying.dot(W3, dz3)
        dz2 = [dz2_pre[i] * relu_derivative(z2)[i] for i in range(len(dz2_pre))]
        dW2 = renumpying.outer_array(a1, dz2)
        db2 = dz2

        # dz1 = dot(W2, dz2) * relu_derivative(z1)
        dz1_pre = renumpying.dot(W2, dz2)
        dz1 = [dz1_pre[i] * relu_derivative(z1)[i] for i in range(len(dz1_pre))]
        dW1 = renumpying.outer_array(x, dz1)
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
z1 = renumpying.outer_array(x_test_vec, W1) + b1
a1 = relu(z1)
z2 = renumpying.outer_array(a1, W2) + b2
a2 = relu(z2)
z3 = renumpying.outer_array(a2, W3) + b3
a3 = relu(z3)
z4 = renumpying.outer_array(a3, W4) + b4
y_pred = softmax(z4)

predicted_move = int(renumpying.argmax(y_pred))
your_move = (predicted_move + 1) % 3

# --- Save weights and biases ---
#all_my_weights_biases = {
    #"layer_1": {"weights": W1.tolist(), "biases": b1.tolist()},
    #"layer_2": {"weights": W2.tolist(), "biases": b2.tolist()},
    #"layer_3": {"weights": W3.tolist(), "biases": b3.tolist()},
    #"layer_4": {"weights": W4.tolist(), "biases": b4.tolist()},
#}

#with open("weights_biases.pkl", "wb") as f:
    #pickle.dump(all_my_weights_biases, f)

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
