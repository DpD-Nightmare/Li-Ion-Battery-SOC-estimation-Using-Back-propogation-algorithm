'''
Data loading filtering and use for training 
NN FFBackprop without stoping criteria.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import convolve

#Importing Data files
fname = "Traing_battery_DST.xlsx"
data= pd.read_excel("Traing_battery_DST.xlsx")
y0 = []
#Define of SMA
def movingavrage(value, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(value, weights,'valid')
    return sma

#defining the activation function for NN
def sigmoid (x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

#data= data.sheet_by_index(0)
a=data['Current(A)'] #Importing Current data from data sheet
curr = movingavrage (a,10) #Filter the data with  window of 10
curr = list(curr) #Declaring the List

v=data['Voltage(V)'] #Imporint Voltage data from data sheet
vol = movingavrage (v,10) #Filtering the data
vol = list(vol)

t=data['Temp(C)'] #Imporitng Temprature data from data sheet
te= movingavrage (t,10) #Filtering the data
te= list(te)

ah=data['cap(Ah)'] #Importing the Charge capacity form data sheet
ch= movingavrage (ah,10) #Filtering the data
ch= list(ch)
yd =[]
ye =[]
for i in range(8085):
    data_in =np.array([curr[i],vol[i],te[i]]) #Creating Input data to the Neural Network I,V,T
    data_in =np.reshape(data_in,(1,3)) #Transpose the NN_input datas
    #expected result
    cha=np.array([ch[i]]) #Converting the list value in to Matrix form
    charge= np.reshape(cha,(1,1)) #Transposed the Matrix for further calculation
    data_ref = np.array(charge) #Creating the Refrance output datas for NN SOC'''
    yd.append(float(data_ref))

    #print('refrance data matrix',data_ref.shape)

    #Giving input and output dataset to NN for training purposes
    inputs = data_in
    expected_output = data_ref
    #number of epochs to train NN
    epochs = 100
    lr=1.5
    inputLayerNeurons, frist_hiddenLayerNeurons, second_hiddenLayerNeurons,  outputLayerNeurons = 3,3,2,1

    #random weights and bias initialization
    hidden1_weights = np.random.uniform(size=(inputLayerNeurons,frist_hiddenLayerNeurons))
    hidden1_bias = np.random.uniform(size=(1,frist_hiddenLayerNeurons))

    hidden2_weights = np.random.uniform(size=(frist_hiddenLayerNeurons,second_hiddenLayerNeurons))
    hidden2_bias = np.random.uniform(size=(1,second_hiddenLayerNeurons))

    output_weights = np.random.uniform(size=(second_hiddenLayerNeurons,outputLayerNeurons))
    output_bias = np.random.uniform(size=(1,outputLayerNeurons))

    #print('')
    #print("Initial hidden weights:",end='')
    #print(*hidden1_weights)
    #print("Initial hidden biases:",end='')
    #print(*hidden1_bias)
    #print('')
    #print("Initial hidden weights:",end='')
    #print(*hidden2_weights)
    #print("Initial hidden biases:",end='')
    #print(*hidden2_bias)
    #print('')
    #print("Initial output weights:",end='')
    #print(*output_weights)
    #print("Initial output biases",end='')
    #print(*output_bias)


    #NN Training algorithm (Backpropagation)
    yee =0
    yes = 0
    for _ in range(epochs):
        #Forward Propagation
        #1st layer
        frist_hidden_layer_activation = np.dot(inputs,hidden1_weights)
        #print(frist_hidden_layer_activation.shape)
        frist_hidden_layer_activation += hidden1_bias
        frist_hidden_layer_output = sigmoid(frist_hidden_layer_activation)
        #print(frist_hidden_layer_output)
        #2nd Layer
        second_hidden_layer_activation = np.dot(frist_hidden_layer_output, hidden2_weights)
        second_hidden_layer_activation += hidden2_bias
        second_hidden_layer_output = sigmoid(second_hidden_layer_activation)
        #Output layer
        output_layer_activation = np.dot(second_hidden_layer_output,output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)
        #print('Expected',expected_output.shape)
        print('pridect',predicted_output.shape,second_hidden_layer_output.shape,output_weights.shape)
    
        #Backpropagation level
        error = expected_output - predicted_output
        yee = predicted_output
        yes = error
        #print(error)
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_second_hidden_layer = d_predicted_output.dot(output_weights.T)
        
        d_hidden2_layer = error_second_hidden_layer * sigmoid_derivative(second_hidden_layer_output)
        error_frist_hidden_layer = d_hidden2_layer.dot(hidden2_weights.T)
        
        d_hidden1_layer = error_frist_hidden_layer * sigmoid_derivative(frist_hidden_layer_output)
    
        #Updating weights and Biases
        output_weights += second_hidden_layer_output.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    
        hidden2_weights += frist_hidden_layer_output.T.dot(d_hidden2_layer) * lr
        hidden2_bias += np.sum(d_hidden2_layer,axis=0,keepdims=True) * lr
    
        hidden1_weights += inputs.T.dot(d_hidden1_layer) * lr
        hidden1_bias += np.sum(d_hidden1_layer,axis=0,keepdims=True) * lr
    
    #print("")
    #print("Final frist hidden weights:",end='')
    #print(*hidden1_weights)
    #print("Final frist hidden bias:",end='')
    #print(*hidden1_bias)
    #print('')
    #print("")
    #print("Final second hidden weights:",end='')
    #print(*hidden2_weights)
    #print("Final second hidden bias:",end='')
    #print(*hidden2_bias)
    #print('')
    #print("Final Output Weights:",end='')
    #print(*output_weights)
    #print("Final Output Bias:",end='')
    #print(*output_bias)
    print(" ieee ",i)
    y0.append(float(yee))
    ye.append(float(yes))
    print("yee:",yee)
print("\nOutput from NN after 10000 epochs:",end='')
print(*predicted_output)
plt.plot(y0)
plt.plot(yd)
plt.plot(ye)
plt.show()

print(y0)