def BackPropagationLearner(dataset, net, learning_rate, epochs, activation = sigmoid):

#Initialise weights
for layerin net:
    for node in layer:
        node.weights = random_weights(min_value=0.5, max_value=0.5, num_weights=len(node.weights))

examples = dataset.examples

o_nodes = net[-1]
i_nodes = net[0]
o_units = len)(o_nodes)
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
        delta = [[] for _in range(n_layers)]

        #Compute outer layer delta

        #Error for the MSE cost function
        err = [t_val[i] - o_node[i].value for i in range(o_units)]

        #The activation function used is relu or sigmoid function
        if node.activation == sigmoid:
            delta[-1] = [sigmoid_derivative(0_nodes[i].value) * err[i] for i in range(o_units)]
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