#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:55:01 2019

@author: dhrupad
"""
'''
from learning import *
from notebook import psource, pseudocode

#psource(NeuralNetLearner)

def NeuralNetLearner (dataset, hidden_layer_sizes=None,
                      learning_rate=0.01, epochs=100,
                      activation=sigmoid):
    """Layered feed-fwd network.
    hidden_layer_sizes: List of number of hidden unit
    per hidden layer
    Learning rate: Learning rate of gradient descent
    epochs: Number of passes over the dataset"""
    
    hidden_layer_sizes = hidden_layer_sizes or [3] # default value
    i_units = len(dataset.input)
    o_units = len(dataset.values[dataset.target])
    
    # construct a network
    raw_net=network(i_units, hidden_layer_sizes,
                    o_units, activation)
    learn_net=BackPropagationLearner(dataset, raw_net, learning_rate,
                                     epochs, activation)
    
    def predict(example):
       #Input nodes
       i_nodes=learned_net[0]
       
       #Activation input layer
       for v, n in zip(example, i_nodes):
           n.value = v
           
       # Forwaed pass
       for layer in learned_net[1:]:
           for node in layer:
               inc = [n.alue for n in node.inputs]
               in_val=dotproduct(inc,node.weights)
               node.value=node.activation(in_val)
               
       #Hypothesis
       o_node = learned_net[-1]
       prediction=find_max_node(o_nodes)
       return prediction
    return predict

'''
   #imlimaintation of backprob
from learning import*
from notebook import psource, pseudocode
psource(NeuralNetLearner)
def BackPropagationLearner(dataset, net, learning_rate, epochs, activation=sigmoid):

    #Initialise weights
    for layer in net:
        for node in layer:
            node.weights = random_weights(min_value=0.5, max_value=0.5, num_weights=len(node.weights))

    examples = dataset.examples

    o_nodes = net[-1]
    i_nodes = net[0]
    o_units = len(o_nodes)
    idx_t = dataset.target
    idx_i = dataset.inputs
    n_layers = len(net)

    inputs, target = init_examples(examples, idx_i, idx_t, o_units)

    for epoch in range(epochs):
    #Iterate over each examples
        for e in rang(len(examples)):
            i_val = inputs[e]
            t_val = targets[e]

        #Activation input layer
        for v,n in zip(i_val, i_nodes):
            n.value = value

        #forward Pass
        for layer in net[1:]:
            for node in layer:
                inc = [n.value for n in node.inputs]
                in_val = dotproduct(inc, node.weights)
                node.value =node.activation(in_val)

        #Initialize dalta
        delta = [[] for _ in range(n_layers)]

        #Compute outer layer delta

        #Error for the MSE cost function
        err = [t_val[i] - o_node[i].value for i in range(o_units)]

        #The activation function used is relu or sigmoid function
        if node.activation == sigmoid:
            delta[-1] = [sigmoid_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
        else:
            delta[-1] = [sigmoid_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]

            #Backward Pass
            h_layers = n_layers -2
            for i in range(h_layers, 0, -1):
                layer = net[i]
                h_units = len(layer)
                nx_layer = net [i+1]

                #weights from each ith layer node to each i+1th layer node
                w= [[node.weights[k] for node in nx_layer] for k in range(h_units)]

                if activation == sigmoid:
                    delta[i] = [sigmoid_derivative(layer[j].value) * dotproduct(w[j], delta[j+1]) for j in range(h_units)]

                #Upadte weights
                for i in range(1, n_layers):
                    layer = net[i]
                    inc = [node.value for node in net[i-1]]
                    units = len(layer)
                    for j in range(units):
                        layer[j].weights = vector_add(layer[j].weights, scalar_vector_product(learning_rate * delta[i][j], inc))

    return net