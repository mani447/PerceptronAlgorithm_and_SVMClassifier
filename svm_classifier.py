#netid:MRM190005
import pandas as pd
import numpy as np
import math
import cvxopt

# read data from file
with open(r"mystery.data",'rb') as file:
    data = pd.read_csv(file, sep=',', header=None)
    
#combinations
def no_of_combinations(n,r):
    fact = math.factorial
    n_c = fact(n) // fact(r) // fact(n-r)
    return n_c

#augmentation
def augmentation_of_features(data):
    t = np.ones((data.shape[0],data.shape[1]*2 +no_of_combinations(data.shape[1],2) + 1))
    for i in range(0,data.shape[0]):
        a = data[i]
        b = np.square(a)
        c = np.append(a,b)
        t_list = [j for j in c]
        for j in range(0,data.shape[1]):
            for k in range(j+1,data.shape[1]):
                t_list.append(data[i][j]*data[i][k])
        t_list.append(1)
        z = np.array(t_list)
        t[i] = z
    return t

#svm classifier
def support_vector_machine_classifier(i_data, o_data):
    if i_data.shape[0] != o_data.shape[0]:
        raise ValueError("Input and Output data size Mismatch")
    datasetsize = i_data.shape[0]
    number_of_inputs = i_data.shape[1]
    c_matrix = np.zeros((datasetsize,number_of_inputs))
    for i in range(0,datasetsize):
        c_matrix[i] = -1* i_data[i] * o_data[i]
    p = np.zeros((number_of_inputs,number_of_inputs))
    np.fill_diagonal(p,1)
    p = cvxopt.matrix(p, tc='d')
    q = cvxopt.matrix(np.zeros((number_of_inputs,1)), tc='d')
    g = cvxopt.matrix(c_matrix, tc='d')
    h = cvxopt.matrix((-1* np.ones((datasetsize,1))),tc='d')
    sol = cvxopt.solvers.qp(p,q,g,h)
    return sol['x']

#loss function
def loss(i_data, o_data, weightvector):
    est_output = np.matmul(weightvector, np.transpose(i_data))
    margin = np.transpose(np.multiply(est_output, np.transpose(o_data)))
    loss = np.sum(np.where(margin < 0))
    return loss

# finding the support vectors
def support_vectors(i_data, o_data, weightvector):
    est_output = np.matmul(weightvector, np.transpose(i_data))
    margin = np.transpose(np.multiply(est_output, np.transpose(o_data)))
    supportvectors = list()
    for i in range(0,len(margin)):
        if (margin[i] > 0.999) & (margin[i] < 1.001):
            supportvectors.append(i_data[i][:4])
    supportvectors = np.array(supportvectors)
    return supportvectors

# main
inputdata = np.array(data[data.columns[0:4]])
temp = np.ones((inputdata.shape[0],inputdata.shape[1]+1))
outputdata = data[4]
data_augmented = augmentation_of_features(inputdata)
w_vector = support_vector_machine_classifier(data_augmented, outputdata)
w_vector_list = [x for x in w_vector]
w_vector = np.array(w_vector_list)
supportvectors = support_vectors(data_augmented,outputdata,w_vector)

# printing weight vector, support vectors, margin and loss
print('weight vector = ' + str(w_vector))
print('Loss = ' + str(loss(data_augmented,outputdata,w_vector)))
print("supportvectors")
print(supportvectors)
print('Margin = '+str(1/np.linalg.norm(w_vector,2)))
