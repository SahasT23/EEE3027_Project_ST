

import numpy as np

# ACTIVATION FUNCTIONS & DERIVATIVES

def sigmoid(x):
    """Sigmoid activation: squashes values to (0, 1).
    Used for the forget, input, and output gates in the LSTM."""
    # Clip to prevent overflow in exp()
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(s):
    """Derivative of sigmoid, given the sigmoid OUTPUT s (not raw input).
    d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))"""
    return s * (1.0 - s)

def tanh(x):
    """Tanh activation: squashes values to (-1, 1).
    Used for candidate cell state and hidden state modulation."""
    return np.tanh(x)

def tanh_derivative(t):
    """Derivative of tanh, given the tanh OUTPUT t (not raw input).
    d/dx tanh(x) = 1 - tanh(x)^2"""
    return 1.0 - t ** 2


# WEIGHT INITIALISATION

def initialise_lstm_weights(input_size, hidden_size):
    """
    Initialise all LSTM gate weights using Xavier/Glorot initialisation.
    
    An LSTM has 4 sets of weights (one per gate):
        - Forget gate (f): decides what to discard from cell state
        - Input gate (i): decides which new values to store
        - Candidate gate (g): creates candidate values to add to cell state
        - Output gate (o): decides what part of cell state to output
    
    Each gate has:
        - W: weight matrix for input x_t          (input_size x hidden_size)
        - U: weight matrix for previous hidden h_{t-1}  (hidden_size x hidden_size)
        - b: bias vector                           (1 x hidden_size)
    
    Args:
        input_size:  dimensionality of input features
        hidden_size: number of LSTM hidden units
    
    Returns:
        params: dictionary containing all weight matrices and biases
    """
    # Xavier initialisation scale factor for stable gradients
    scale_w = np.sqrt(2.0 / (input_size + hidden_size))
    scale_u = np.sqrt(2.0 / (hidden_size + hidden_size))

    params = {}
    for gate in ['f', 'i', 'g', 'o']:  # forget, input, candidate, output
        params[f'W_{gate}'] = np.random.randn(input_size, hidden_size) * scale_w
        params[f'U_{gate}'] = np.random.randn(hidden_size, hidden_size) * scale_u
        params[f'b_{gate}'] = np.zeros((1, hidden_size))

    # --- Dense output layer weights ---
    # Maps from hidden state to a single regression output
    scale_dense = np.sqrt(2.0 / (hidden_size + 1))
    params['W_dense'] = np.random.randn(hidden_size, 1) * scale_dense
    params['b_dense'] = np.zeros((1, 1))

    return params

# LSTM FORWARD PASS (Single Timestep)

def lstm_step_forward(x_t, h_prev, c_prev, params):
    """
    Args:
        x_t:    input at time t              (batch_size, input_size)
        h_prev: previous hidden state        (batch_size, hidden_size)
        c_prev: previous cell state          (batch_size, hidden_size)
        params: weight dictionary
    
    Returns:
        h_t:   new hidden state
        c_t:   new cell state
        cache: intermediate values needed for backpropagation
    """
    # --- Forget Gate: what fraction of old cell state to keep ---
    f_t = sigmoid(x_t @ params['W_f'] + h_prev @ params['U_f'] + params['b_f'])

    # --- Input Gate: what fraction of new candidate to add ---
    i_t = sigmoid(x_t @ params['W_i'] + h_prev @ params['U_i'] + params['b_i'])

    # --- Candidate Gate: proposed new values for cell state ---
    g_t = tanh(x_t @ params['W_g'] + h_prev @ params['U_g'] + params['b_g'])

    # --- Output Gate: what fraction of cell state to expose as hidden state ---
    o_t = sigmoid(x_t @ params['W_o'] + h_prev @ params['U_o'] + params['b_o'])

    # --- Update cell state: forget old + add new ---
    c_t = f_t * c_prev + i_t * g_t

    # --- Compute new hidden state ---
    tanh_c_t = tanh(c_t)
    h_t = o_t * tanh_c_t

    # Store everything needed for the backward pass
    cache = {
        'x_t': x_t, 'h_prev': h_prev, 'c_prev': c_prev,
        'f_t': f_t, 'i_t': i_t, 'g_t': g_t, 'o_t': o_t,
        'c_t': c_t, 'tanh_c_t': tanh_c_t, 'h_t': h_t
    }
    return h_t, c_t, cache


# 
# FULL FORWARD PASS (All Timesteps + Dense Output)
# 

def forward_pass(X, params, hidden_size):
    """
    Args:
        X:           input sequence (batch_size, seq_length, input_size)
        params:      weight dictionary
        hidden_size: number of LSTM hidden units
    
    Returns:
        y_pred:  regression predictions (batch_size, 1)
        caches:  list of cache dicts from each timestep (for backprop)
        h_t:     final hidden state
        c_t:     final cell state
    """
    batch_size, seq_length, input_size = X.shape

    # Initialise hidden state and cell state to zeros
    h_t = np.zeros((batch_size, hidden_size))
    c_t = np.zeros((batch_size, hidden_size))

    caches = []

    # Process each timestep sequentially
    for t in range(seq_length):
        x_t = X[:, t, :]  # Extract input at timestep t: (batch_size, input_size)
        h_t, c_t, cache = lstm_step_forward(x_t, h_t, c_t, params)
        caches.append(cache)

    # --- Dense layer: map final hidden state to regression output ---
    # y = h_T @ W_dense + b_dense
    y_pred = h_t @ params['W_dense'] + params['b_dense']

    return y_pred, caches, h_t, c_t

# LOSS FUNCTION 

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error loss for regression.
    L = (1/N) * sum((y_pred - y_true)^2)
    
    Returns:
        loss:      scalar MSE value
        d_y_pred:  gradient of loss w.r.t. y_pred (batch_size, 1)
    """
    batch_size = y_pred.shape[0]
    error = y_pred - y_true
    loss = np.mean(error ** 2)
    # Gradient: dL/dy_pred = 2*(y_pred - y_true) / N
    d_y_pred = 2.0 * error / batch_size
    return loss, d_y_pred

# LSTM BACKWARD PASS (Single Timestep)

def lstm_step_backward(d_h_next, d_c_next, cache, params):
    """
    Args:
        d_h_next: gradient flowing into h_t    (batch_size, hidden_size)
        d_c_next: gradient flowing into c_t    (batch_size, hidden_size)
        cache:    stored forward pass values
        params:   weight dictionary
    
    Returns:
        d_h_prev: gradient to pass to h_{t-1}
        d_c_prev: gradient to pass to c_{t-1}
        grads:    dictionary of parameter gradients for this timestep
    """
    # Unpack cached values
    x_t = cache['x_t']
    h_prev = cache['h_prev']
    c_prev = cache['c_prev']
    f_t = cache['f_t']
    i_t = cache['i_t']
    g_t = cache['g_t']
    o_t = cache['o_t']
    c_t = cache['c_t']
    tanh_c_t = cache['tanh_c_t']

    # --- Gradient through h_t = o_t * tanh(c_t) ---
    # d_o_t: gradient w.r.t. output gate
    d_o_t = d_h_next * tanh_c_t
    # d_tanh_c_t: gradient flowing into tanh(c_t) from h_t
    d_tanh_c_t = d_h_next * o_t

    # --- Gradient through cell state ---
    # tanh(c_t) feeds into h_t, so gradient passes through tanh derivative
    d_c_t = d_c_next + d_tanh_c_t * tanh_derivative(tanh_c_t)

    # --- Gradient through c_t = f_t * c_prev + i_t * g_t ---
    d_f_t = d_c_t * c_prev          # forget gate gradient
    d_i_t = d_c_t * g_t             # input gate gradient
    d_g_t = d_c_t * i_t             # candidate gradient
    d_c_prev = d_c_t * f_t          # gradient flowing to previous cell state

    # --- Gradient through gate activations (chain rule with activation derivatives) ---
    # For sigmoid gates: d_raw = d_gate * sigmoid'(gate) = d_gate * gate * (1 - gate)
    d_f_raw = d_f_t * sigmoid_derivative(f_t)
    d_i_raw = d_i_t * sigmoid_derivative(i_t)
    d_o_raw = d_o_t * sigmoid_derivative(o_t)
    # For tanh candidate: d_raw = d_g * tanh'(g) = d_g * (1 - g^2)
    d_g_raw = d_g_t * tanh_derivative(g_t)

    # --- Compute parameter gradients for each gate ---
    # Each gate's pre-activation: z = x_t @ W + h_prev @ U + b
    # So: dW = x_t^T @ d_raw, dU = h_prev^T @ d_raw, db = sum(d_raw)
    grads = {}
    for gate, d_raw in [('f', d_f_raw), ('i', d_i_raw), ('g', d_g_raw), ('o', d_o_raw)]:
        grads[f'W_{gate}'] = x_t.T @ d_raw
        grads[f'U_{gate}'] = h_prev.T @ d_raw
        grads[f'b_{gate}'] = np.sum(d_raw, axis=0, keepdims=True)

    # --- Gradient flowing back to previous hidden state ---
    # h_prev contributes to all 4 gates via U matrices
    d_h_prev = (d_f_raw @ params['U_f'].T +
                d_i_raw @ params['U_i'].T +
                d_g_raw @ params['U_g'].T +
                d_o_raw @ params['U_o'].T)

    return d_h_prev, d_c_prev, grads
 
# FULL BACKWARD PASS (Backpropagation Through Time)

def backward_pass(d_y_pred, caches, params, hidden_size):
    """
    Full backward pass: backpropagate from the loss through the dense layer
    and then through all LSTM timesteps (BPTT).
    
    Args:
        d_y_pred:    gradient of loss w.r.t. predictions (batch_size, 1)
        caches:      list of cache dicts from forward pass
        params:      weight dictionary
        hidden_size: number of LSTM hidden units
    
    Returns:
        all_grads: dictionary of accumulated gradients for all parameters
    """
    seq_length = len(caches)
    h_final = caches[-1]['h_t']

    # --- Backprop through dense layer: y = h_T @ W_dense + b_dense ---
    all_grads = {}
    all_grads['W_dense'] = h_final.T @ d_y_pred       # (hidden_size, 1)
    all_grads['b_dense'] = np.sum(d_y_pred, axis=0, keepdims=True)  # (1, 1)

    # Gradient flowing into the final hidden state from the dense layer
    d_h = d_y_pred @ params['W_dense'].T  # (batch_size, hidden_size)

    # No gradient flows into cell state from the dense layer
    d_c = np.zeros_like(d_h)

    # --- Initialise accumulated gradients for LSTM parameters ---
    for gate in ['f', 'i', 'g', 'o']:
        all_grads[f'W_{gate}'] = np.zeros_like(params[f'W_{gate}'])
        all_grads[f'U_{gate}'] = np.zeros_like(params[f'U_{gate}'])
        all_grads[f'b_{gate}'] = np.zeros_like(params[f'b_{gate}'])

    # --- Backpropagate through time (reverse order) ---
    for t in reversed(range(seq_length)):
        d_h, d_c, step_grads = lstm_step_backward(d_h, d_c, caches[t], params)

        # Accumulate gradients across all timesteps
        for key in step_grads:
            all_grads[key] += step_grads[key]

    # --- Gradient clipping to prevent exploding gradients ---
    # This is critical for LSTM training stability
    max_grad_norm = 5.0
    total_norm = 0.0
    for key in all_grads:
        total_norm += np.sum(all_grads[key] ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_grad_norm:
        scale = max_grad_norm / total_norm
        for key in all_grads:
            all_grads[key] *= scale

    return all_grads

# ADAM OPTIMISER

class AdamOptimiser:
    """
    Update rule:
        m = beta1 * m + (1 - beta1) * grad          # First moment (mean)
        v = beta2 * v + (1 - beta2) * grad^2         # Second moment (variance)
        m_hat = m / (1 - beta1^t)                     # Bias correction
        v_hat = v / (1 - beta2^t)                     # Bias correction
        param -= lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep counter for bias correction

        # Initialise first and second moment estimates to zero
        self.m = {key: np.zeros_like(val) for key, val in params.items()}
        self.v = {key: np.zeros_like(val) for key, val in params.items()}

    def step(self, params, grads):
        """Perform one optimization step, updating all parameters in-place."""
        self.t += 1
        for key in params:
            # Update biased first moment estimate (momentum)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second moment estimate (adaptive lr)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2

            # Bias-corrected estimates (important in early steps)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

# DATA GENERATION (Synthetic Sine Wave Regression Task)

def generate_sine_data(n_samples=1000, seq_length=20, noise=0.05):
    """
    Args:
        n_samples:  number of (sequence, target) pairs
        seq_length: number of timesteps per input sequence
        noise:      standard deviation of Gaussian noise added
    
    Returns:
        X: input sequences  (n_samples, seq_length, 1)
        y: target values    (n_samples, 1)
    """
    # Generate a long sine wave with some noise
    t = np.linspace(0, 4 * np.pi, n_samples + seq_length)
    signal = np.sin(t) + np.random.randn(len(t)) * noise

    X = []
    y = []
    for i in range(n_samples):
        # Input: seq_length consecutive values
        X.append(signal[i:i + seq_length])
        # Target: the very next value
        y.append(signal[i + seq_length])

    X = np.array(X).reshape(n_samples, seq_length, 1)  # Add feature dimension
    y = np.array(y).reshape(n_samples, 1)
    return X, y

# TRAINING LOOP

def train(X_train, y_train, X_val, y_val,
          hidden_size=32, epochs=100, batch_size=32, lr=0.001):
    """
    Train the LSTM regression model.
    
    Args:
        X_train, y_train: training data
        X_val, y_val:     validation data
        hidden_size:      number of LSTM hidden units
        epochs:           number of full passes through training data
        batch_size:       number of samples per gradient update
        lr:               learning rate for Adam optimizer
    
    Returns:
        params:       trained weight dictionary
        train_losses: list of training losses per epoch
        val_losses:   list of validation losses per epoch
    """
    n_samples, seq_length, input_size = X_train.shape

    # Initialise weights and optimizer
    params = initialise_lstm_weights(input_size, hidden_size)
    optimizer = AdamOptimiser(params, lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Shuffle training data each epoch for better generalization
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        n_batches = 0

        # --- Mini-batch training ---
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass: compute predictions
            y_pred, caches, _, _ = forward_pass(X_batch, params, hidden_size)

            # Compute loss
            loss, d_y_pred = mse_loss(y_pred, y_batch)
            epoch_loss += loss
            n_batches += 1

            # Backward pass: compute gradients via BPTT
            grads = backward_pass(d_y_pred, caches, params, hidden_size)

            # Update weights using Adam
            params = optimizer.step(params, grads)

        # --- Epoch-level logging ---
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation loss (no gradient computation needed)
        val_pred, _, _, _ = forward_pass(X_val, params, hidden_size)
        val_loss, _ = mse_loss(val_pred, y_val)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")

    return params, train_losses, val_losses

# PREDICTION

def predict(X, params, hidden_size):
    """Run a forward pass and return predictions (no gradients needed)."""
    y_pred, _, _, _ = forward_pass(X, params, hidden_size)
    return y_pred

# MAIN: Putting It All Together

if __name__ == "__main__":
    np.random.seed(42)

    # --- Hyperparameters ---
    HIDDEN_SIZE = 32      # Number of LSTM units
    SEQ_LENGTH = 25       # Input sequence length (lookback window)
    EPOCHS = 100          # Training epochs
    BATCH_SIZE = 32       # Mini-batch size
    LEARNING_RATE = 0.005 # Adam learning rate

    # --- Generate Data ---
    print("Generating synthetic sine wave data...")
    X, y = generate_sine_data(n_samples=2000, seq_length=SEQ_LENGTH, noise=0.02)

    # Train/validation/test split (70/15/15)
    n_train = int(0.7 * len(X))
    n_val = int(0.85 * len(X))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Sequence length: {SEQ_LENGTH} | Hidden size: {HIDDEN_SIZE}")
    print("-" * 60)

    # --- Train ---
    params, train_losses, val_losses = train(
        X_train, y_train, X_val, y_val,
        hidden_size=HIDDEN_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )

    # --- Evaluate on Test Set ---
    print("-" * 60)
    y_pred_test = predict(X_test, params, HIDDEN_SIZE)
    test_loss, _ = mse_loss(y_pred_test, y_test)
    print(f"Final Test MSE: {test_loss:.6f}")

    # Show a few sample predictions vs actual
    print("\nSample Predictions vs Ground Truth:")
    print(f"{'Predicted':>12} | {'Actual':>12} | {'Error':>12}")
    print("-" * 42)
    for i in range(min(10, len(y_test))):
        pred = y_pred_test[i, 0]
        actual = y_test[i, 0]
        print(f"{pred:12.6f} | {actual:12.6f} | {pred - actual:12.6f}")