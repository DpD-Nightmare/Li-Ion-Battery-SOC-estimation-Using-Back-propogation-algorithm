#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:20:05 2019

@author: dhrupad
"""
import numpy as np
#np.random.seed(0)

#defining the activation function for NN
def sigmoid (x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

#Giving input and output dataset to NN for training purposes
inputs =np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])
#number of epochs to train NN
epochs = 10000
lr=0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1,hiddenLayerNeurons))

output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

print("Initial hidden weights:",end='')
print(*hidden_weights)
print("Initial hidden biases:",end='')
print(*hidden_bias)

print("Initial output weights:",end='')
print(*output_weights)
print("Initial output biases",end='')
print(*output_bias)


#NN Training algorithm (Backpropagation)

for _ in range(epochs):
    #Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    #Backpropagation level
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    #Updating weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
    
print("")
print("Final hidden weights:",end='')
print(*hidden_weights)
print("Final hidden bias:",end='')
print(*hidden_bias)

print("Final Output Weights:",end='')
print(*output_weights)
print("Final Output Bias:",end='')
print(*output_bias)

print("\nOutput from NN after 10000 epochs:",end='')
print(*predicted_output)