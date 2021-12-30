import numpy as np
import re

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weight:\n', synaptic_weights)

for i in range(20000):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments) 


print('Synaptic weights after training:\n', synaptic_weights)

print('Output:\n', outputs)

userInput = input('New input: (3 separated 1s or 0s): ')

userInputValue = re.findall('\d', userInput)

valid = True

if (len(userInputValue) > 3): 
    valid = False
else: 
    for i in range(len(userInputValue)):
        userInputValue[i] = float(userInputValue[i])
        if not (userInputValue[i] in [0, 1]): 
            valid = False
            break
    
if valid:
    userOutput = sigmoid(np.dot(userInputValue, synaptic_weights))
    print('Output: ', userOutput)
else: 
    print('Input invalid!')