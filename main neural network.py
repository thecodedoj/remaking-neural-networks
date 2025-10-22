import os, json,pickle , renumpying
from renumpying import sqroot, sum_func, exp, zero
from random_number_generators import lcg , uniformity , randint
import redoing_numpy_arrays as rna
import remaking_the_logarithm as rtl
# Control verbose debug output
VERBOSE = True

# --- Numerical sanitizer (no math module) ---
def sanitize_vector(v, clip_min=-50.0, clip_max=50.0):
    """Replace NaN/Inf with 0 and clip values to [clip_min, clip_max]."""
    out = []
    for x in v:
        # detect NaN: NaN != NaN
        try:
            if x != x:
                x = 0.0
        except Exception:
            x = 0.0
        # detect huge values (treated as inf)
        try:
            if abs(x) > 1e300:
                x = 0.0
        except Exception:
            x = 0.0
        # clip
        if x < clip_min:
            x = clip_min
        if x > clip_max:
            x = clip_max
        out.append(float(x))
    return out

# Elementwise vector add helper (handles Array wrappers)
def vec_add(a, b):
    if hasattr(a, 'data'): a = a.data
    if hasattr(b, 'data'): b = b.data
    return [a[i] + b[i] for i in range(len(a))]
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
    if VERBOSE:
        print(f"ReLU input type: {type(x)}")  # Debug print
    # Convert Array to list
    if hasattr(x, 'data'):
        x = x.data
        if VERBOSE:
            print(f"ReLU extracted data type: {type(x)}")  # Debug print
    
    # Ensure x is a flat list
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        x = [item for sublist in x for item in sublist]  # flatten one level
        
    if VERBOSE:
        print(f"ReLU x before: {x[:5]}...")  # Show first few elements
    result = [max(0, float(xi)) for xi in x]  # Convert to float to ensure consistency
    if VERBOSE:
        print(f"ReLU input shape: {len(x)}, output shape: {len(result)}")
        print(f"ReLU result first few elements: {result[:5]}...")  # Show first few elements
    return result

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
    # Ensure x is a plain list (extract data if necessary)
    if hasattr(x, 'data'):
        x = x.data
    # If x is a nested list (e.g., shape (n,1)), flatten one level
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        x = [item for sublist in x for item in sublist]

    # Numerically stable softmax for lists
    m = max(x)
    exps = [exp(xi - m) for xi in x]
    s = sum_func(exps)
    if s == 0:
        # fallback to uniform distribution to avoid division by zero
        return [1.0 / len(exps)] * len(exps)
    return [e / s for e in exps]

# --- Xavier initialization ---
def xavier_init(size_in, size_out):
    limit = sqroot(6 / (size_in + size_out))
    # Initialize with correct orientation from the start
    weights = uniformity(-limit, limit, size=(size_in, size_out), seed_start=1)
    return weights


# --- Initialize weights and biases ---
def print_shape(matrix, name):
    if isinstance(matrix[0], list):
        print(f"{name} shape: ({len(matrix)}, {len(matrix[0])})")
    else:
        print(f"{name} shape: ({len(matrix)},)")

W1 = xavier_init(input_size, hidden1_size)
if VERBOSE: print_shape(W1, "W1")
b1 = zero(hidden1_size)
W2 = xavier_init(hidden1_size, hidden2_size)
if VERBOSE: print_shape(W2, "W2")
b2 = zero(hidden2_size)
W3 = xavier_init(hidden2_size, hidden3_size)
if VERBOSE: print_shape(W3, "W3")
b3 = zero(hidden3_size)
W4 = xavier_init(hidden3_size, output_size)
if VERBOSE: print_shape(W4, "W4")
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
# NOTE: lower these while debugging; restore later
num_samples = 50
X = []
Y = renumpying.zero((num_samples, 3))
for i in range(num_samples):
    x_vec, moves_hist = generate_random_rps_input(history_length)
    X.append(x_vec)
    move = randint(0,3,1)[0]
    Y[i][move] = 1
X = rna.Array(X)

# --- Cross-entropy loss ---
def cross_entropy(y_true, y_pred):
    # Ensure lists
    if hasattr(y_true, 'data'):
        y_true = y_true.data
    if hasattr(y_pred, 'data'):
        y_pred = y_pred.data
    # Flatten nested lists one level
    if isinstance(y_true, list) and len(y_true) > 0 and isinstance(y_true[0], list):
        y_true = [item for sub in y_true for item in sub]
    if isinstance(y_pred, list) and len(y_pred) > 0 and isinstance(y_pred[0], list):
        y_pred = [item for sub in y_pred for item in sub]

    eps = 1e-8
    vals = []
    for yt, yp in zip(y_true, y_pred):
        vals.append(yt * rtl.my_log(yp + eps))
    return -sum_func(vals)

# --- Training loop ---
epochs = 5
learning_rate = 0.05

for epoch in range(epochs):
    total_loss = 0
    for i in range(num_samples):
        x = X[i]
        y = Y[i]

        # --- Forward pass ---
        # Forward pass with explicit vector addition (use global helper)
        z1 = sanitize_vector(vec_add(renumpying.dot(x, W1), b1))
        a1 = relu(z1)
        z2 = sanitize_vector(vec_add(renumpying.dot(a1, W2), b2))
        a2 = relu(z2)
        z3 = sanitize_vector(vec_add(renumpying.dot(a2, W3), b3))
        a3 = relu(z3)
        z4 = sanitize_vector(vec_add(renumpying.dot(a3, W4), b4))
        print(f"z4 type: {type(z4)}, shape/length: {len(z4)}, first few elements: {z4[:3]}")
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


        # --- Gradient updates (elementwise to handle plain python lists) ---
        def update_weights(W, dW, lr):
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] -= lr * dW[i][j]

        def update_bias(b, db, lr):
            for i in range(len(b)):
                b[i] -= lr * db[i]

        update_weights(W4, dW4, learning_rate)
        update_bias(b4, db4, learning_rate)
        update_weights(W3, dW3, learning_rate)
        update_bias(b3, db3, learning_rate)
        update_weights(W2, dW2, learning_rate)
        update_bias(b2, db2, learning_rate)
        update_weights(W1, dW1, learning_rate)
        update_bias(b1, db1, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/num_samples:.4f}")

# --- Test prediction ---
x_test_vec, x_test_history = generate_random_rps_input(history_length)
z1 = sanitize_vector(vec_add(renumpying.dot(x_test_vec, W1), b1))
a1 = relu(z1)
z2 = sanitize_vector(vec_add(renumpying.dot(a1, W2), b2))
a2 = relu(z2)
z3 = sanitize_vector(vec_add(renumpying.dot(a2, W3), b3))
a3 = relu(z3)
z4 = sanitize_vector(vec_add(renumpying.dot(a3, W4), b4))
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
