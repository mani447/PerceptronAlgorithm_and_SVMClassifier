
#netid:MRM190005
import numpy as np
import pandas as pd

def getloss(input_data, output_data, weight_vector):
    estimated_output = np.matmul(weight_vector, np.transpose(input_data))
    margin_vector = np.transpose(np.multiply(estimated_output, np.transpose(output_data)))
    loss = np.sum(np.where(margin_vector < 0))
    return loss

def gradient_descent(dataX, dataY, step_size=1, threshold=0.001):
    num_of_data_points = dataX.shape[0]
    num_of_input_variables = dataX.shape[1]
    W = np.zeros(num_of_input_variables)
    b = np.ones(num_of_input_variables)
    gradient = np.ones(num_of_input_variables)
    gradientb = np.ones(num_of_input_variables)
    iter = 0
    #calclating perceptron loss for W
    while np.linalg.norm(gradient) > threshold:
        gradient = np.zeros(num_of_input_variables)
        hypothesis = np.dot(W,np.transpose(dataX))
        loss_vector = np.transpose(np.multiply(hypothesis, np.transpose(dataY)))
        for i in range(0,num_of_data_points):
            if (1-loss_vector[i]) >=0:
                gradient = np.add(gradient, -2*(1-loss_vector[i])*dataY[i]*dataX[i])
                
        W = np.add(W, -1*step_size*gradient)
        iter+=1
     
    iter = 0  
    #calclating perceptron loss for b
    while np.linalg.norm(gradientb) > threshold:
        gradientb = np.zeros(num_of_input_variables)
        hypothesis = np.dot(W,np.transpose(dataX))
        loss_vector = np.transpose(np.multiply(hypothesis, np.transpose(dataY)))
        for i in range(0,num_of_data_points):
            if (1-loss_vector[i]) >=0:
                gradientb = np.add(gradientb, -2*(1-loss_vector[i])*dataY[i])
                
        b = np.add(b, -1*step_size*gradientb)
        iter+=1
        if(iter > 1000000):
            break
    
    return (W, b, iter)
	

def stochastic_gradient_descent(dataX, dataY, step_size=0.0007, threshold=0.001):
    num_of_data_points = dataX.shape[0]
    num_of_input_variables = dataX.shape[1]
    W = np.zeros(num_of_input_variables)
    b = np.zeros(num_of_input_variables)
    iter = 0
    lossval=1
    while lossval>0:
        hypothesis = np.dot(dataX[iter%num_of_data_points],W)
        loss = 1-dataY[iter%num_of_data_points]*hypothesis
        if loss>=0:
            gradient = -2*loss*dataY[iter%num_of_data_points]*dataX[iter%num_of_data_points]
        W = W - step_size*gradient
        iter+=1
        lossval = getloss(dataX,dataY,W)
        
    while lossval>0:
        hypothesis = np.dot(dataX[iter%num_of_data_points],W)
        loss = 1-dataY[iter%num_of_data_points]*hypothesis
        if loss>=0:
            gradientb = -2*loss*dataY[iter%num_of_data_points]
        b = b - step_size*gradientb
        iter+=1
        lossval = getloss(dataX,dataY,W)
        
    return (W,b,iter) 
	
	
data = pd.read_csv("perceptron.data",header=None)

dataX = np.array(data[data.columns[0:4]])
dataY = data[4]

weights , b, iterations = gradient_descent(dataX, dataY,step_size=1)
print(weights, b, iterations)

weights_sgd ,b_sgd, iterations_sgd = stochastic_gradient_descent(dataX, dataY,step_size=1)
print(weights_sgd,b_sgd, iterations_sgd)