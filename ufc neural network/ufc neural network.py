import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
        
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
class Activation_Softmax: 
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
   
# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
        
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss): 
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
            # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # Use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# RMSprop optimizer
class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=1e-5, epsilon=1e-7,rho=0.8):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Extract data using panda
df = pd.read_csv(r'C:\Users\mattf\OneDrive\Desktop\cod\ufc results\preprocessed_data.csv')

red = []
blue = []

# Loop over col titles, if first 2 letters are B_ or R_ append to list
for x in df:
    if "B_" in x[0:2]:
        blue.append(x)
    if "R_" in x[0:2]:
        red.append(x)

# Function that takes a number for an argument, then initialises a list and
# appends the data element specified within that col heading
def dataRed(n):
    dataRed = []
    for z in red:
        dataRed.append(df[z][n])     
    return dataRed

def dataBlue(n):
    dataBlue = []
    for z in blue:
        dataBlue.append(df[z][n])
    return dataBlue

# Function that when given a fight number will return the result, as 1 for red and 0 for blue
def outcome(number):
    if df["Winner"][number] == "Red":
        return 1
    else: 
        return 0

data = []
test_result = []

unseen_up2 = 10#/5902 How many of the most recent fights out of 5902
                #do you want to predict

# Read processed data and format it
for j in reversed(range(unseen_up2, 5902)):
    blue_data = np.array(dataBlue(j))  # Convert dataBlue(j) to a NumPy array
    red_data = np.array(dataRed(j))    # Convert dataRed(j) to a NumPy array
    data.append(np.concatenate((blue_data, red_data)))  # Append the concatenated array to the data list
    test_result.append(outcome(j))  # Append the outcome to the test_result list

# Read data and create data that the NN won't see until after training
unseen_data = []
unseen_result=[]
for d in reversed(range(unseen_up2)):
    blue_data = np.array(dataBlue(d))  # Convert dataBlue(j) to a NumPy array
    red_data = np.array(dataRed(d)) 
    unseen_data.append(np.concatenate((blue_data, red_data)))  # Append the concatenated array to the data list
    unseen_result.append(outcome(d))
unseen_data = np.array(unseen_data)
unseen_result = np.array(unseen_result)

# Convert the data and test_result lists to NumPy arrays if needed
X = np.array(data)
y = np.array(test_result)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(144, 256)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(256, 256)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(256,256)
activation3 = Activation_ReLU()

dense4 = Layer_Dense(256,2)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_RMSprop(decay=1e-4)

# Train in loop
for epoch in range(2001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    dense4.forward(activation3.output)
    
    # Takes the output of final dense layer here and returns loss
    loss = loss_activation.forward(dense4.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    # How close are your result guesses to the actual result? 
    accuracy = np.mean(predictions==y) 

    # Print info every 100 epochs
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f} ' )#+ f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_activation.backward(loss_activation.output, y)

    dense4.backward(loss_activation.dinputs)
    activation3.backward(dense4.dinputs)

    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)

    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)

    dense1.backward(activation1.dinputs)

    #update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense4)
    optimizer.post_update_params()

# After training, test the neural network on unseen data
# Perform a forward pass through the neural network layers
dense1.forward(unseen_data)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

dense3.forward(activation2.output)
activation3.forward(dense3.output)

dense4.forward(activation3.output)

# Perform a forward pass through the activation/loss function

loss_activation.forward(dense4.output, unseen_result)

# Get predictions for unseen data
predictions_unseen = np.argmax(loss_activation.output, axis=1)

# Print unseen results and predictions for debugging
print("Unseen Result:", unseen_result)
print("Predictions Unseen:", predictions_unseen)

# Calculate accuracy for unseen data
accuracy_unseen = np.mean(predictions_unseen == unseen_result)

# Print accuracy for unseen data
print(f'Unseen data accuracy: {accuracy_unseen:.3f}')
