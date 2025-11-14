import numpy as np

class Neuron:
    def __init__(self,n_input):
        self.weights = np.random.randn(n_input)
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.dweights = np.zeros_like(self.weights)
        self.dbias = 0


    def activate(self, x):
        # versión numéricamente estable de la sigmoide
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def derivate_activate(self, x):
        return x*(1 - x)
    
    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output
 
    def backward(self, d_output, learning_rate):
        # d_output: gradient of loss w.r.t. this neuron's output (scalar)
        d_activation = d_output * self.derivate_activate(self.output)
        # gradient w.r.t. weights: input_i * d_activation (element-wise)
        self.dweights = self.inputs * d_activation
        # gradient w.r.t. bias is the activation gradient (scalar)
        self.dbias = d_activation
        # gradient w.r.t. inputs (to propagate to previous layer): weight_i * d_activation
        d_input = self.weights * d_activation
        # update parameters (SGD)
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias
        return d_input

#if __name__ == "__main__":
#    neuron = Neuron(3)
#    inputs = np.array([1, 2, 3])
#    output = neuron.forward(inputs)
#    print("Neuron output:", output)

class Layer:

    def __init__(self, num_neurons, inputs_size):
        
        self.neurons = [Neuron(inputs_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
         
    def backward(self, d_outputs, learning_rate):
        # d_outputs: array of gradients for each neuron in this layer
        # initialize d_inputs with same shape as a neuron's inputs
        d_inputs = np.zeros_like(self.neurons[0].inputs)
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs



#if __name__ == "__main__":
#    
#    layer = Layer(3, 4)
#    inputs = np.array([1, 8, 5, 6])
#
#    layer_output = layer.forward(inputs)
#
#    print("Layer output:", layer_output)

class NeuronalNetwork:
    def __init__(self):
        self.layers = []
        self.loss_list = []
    
    def add_layer(self, num_neurons, input_size):
        if not self.layers:
            self.layers.append(Layer(num_neurons, input_size))
        else:
            prev_layer_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, prev_layer_size))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.1,
              print_every=100, early_stopping=False, tol=1e-6, patience=10,
              normalize_inputs=False):
        # Ensure numpy arrays and shapes
        X = np.array(X)
        y = np.array(y)
        # Ensure y shape is (n_samples, n_outputs)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if normalize_inputs:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_std[X_std == 0] = 1.0
            X = (X - X_mean) / X_std

        best_loss = float('inf')
        wait = 0

        for epoch in range(epochs):
            loss = 0.0
            for i in range(len(X)):
                output = self.forward(X[i])
                # ensure output is array-like
                output = np.array(output)
                target = y[i]
                loss += np.mean((target - output) ** 2)
                # gradient of MSE w.r.t. output
                loss_gradient = -2 * (target - output) / output.size
                # detect NaN/Inf in gradient
                if np.any(np.isnan(loss_gradient)) or np.any(np.isinf(loss_gradient)):
                    print(f"NaN/Inf detected in loss gradient at sample {i}, epoch {epoch}")
                    return
                self.backward(loss_gradient, learning_rate)

            loss /= len(X)
            self.loss_list.append(loss)

            # diagnostics
            if np.isnan(loss) or np.isinf(loss):
                print(f"Training stopped: loss is NaN/Inf at epoch {epoch}")
                break

            if epoch % print_every == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch}, loss: {loss}")

            # early stopping
            if early_stopping:
                if best_loss - loss > tol:
                    best_loss = loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}. Best loss: {best_loss}")
                        break

        return self.loss_list



    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predictions.append(self.forward(X[i]))
        # return stacked predictions
        return np.array(predictions)
    

if __name__ == "__main__":
    # Example of usage:
    # Create a neural network with 3 input features, 1 hidden layers, and 1 output layer
    nn = NeuronalNetwork()

    # Add layers: input size for first hidden layer is 3 (e.g., 3 input features)
    nn.add_layer(num_neurons=3, input_size=3)  # First hidden layer with 5 neurons
    nn.add_layer(num_neurons=3, input_size=3)  # Second hidden layer with 4 neurons
    nn.add_layer(num_neurons=1, input_size=4)  # Output layer with 1 neuron

    # Dummy training data (X: input, y: target)
    X = np.array([[0.5, 0.2, 0.1],
                  [0.9, 0.7, 0.3],
                  [0.4, 0.5, 0.8]])
    y = np.array([[0.3, 0.6, 0.9]]).T

    # Train the network
    nn.train(X, y, epochs=4000, learning_rate=0.5)


    # Predict using the trained network
    predictions = nn.predict(X)
    print(f"Predictions: {predictions}")


