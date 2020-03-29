######################################################################
# This module was developed by Harish A and Saksham Jhawar
# This module is a part of Improving Iot edge computing
# Further modules have to be added accordingly.
# First build 21st January 2019
######################################################################

#Importing Libraries##################################################
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import f1_score
from math import exp

#Variable initialisation##############################################
W = np.zeros((2, 1))
b = np.zeros((1,1))
learning_rate = 0.001
inputList = []
outputList = []

#Sigmoid Function#####################################################
def sigmoid(Z):
	return 1/(1+np.e**(-Z))

#Loss Calculation#####################################################
def logistic_loss(y, y_hat):
	return -np.mean(y * np.log(y_hat) + (1-y) * (np.log(1- y_hat)))

#Loading fire positive data###########################################
with open('weatherHistory.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for row in csv_reader:
		if not row[0]=='Formatted Date':
			if row[1]=='Overcast':
				inputList.append([float(row[3]),float(row[5])])
				outputList.append(1)
			elif row[1]=='Clear':
				inputList.append([float(row[3]),float(row[5])])
				outputList.append(0)
				
X = np.array(inputList)
y = np.array(outputList)

#Randomly merging data################################################
perm = np.random.permutation(1118)

#Dividing data into test and train####################################
x_train, x_test = X[perm][224:], X[perm][:224]
y_train, y_test = y[perm][224:], y[perm][:224]
y_train = y_train.reshape(-1,1)
y_test =  y_test.reshape(-1,1)
m = len(y_train)				#Number of samples in training set
#-----------------------------print(x_test,x_train,y_test,y_train)#----When can I delete this?

#Training begins!#####################################################
for epoch in range(5000):
    Z = np.matmul(x_train, W) + b
    A = sigmoid(Z)
    loss = logistic_loss(y_train, A)
    dz = A - y_train
    dw = 1/m * np.matmul(x_train.T, dz)
    db = np.sum(dz)
    
    W = W - learning_rate * dw
    b = b - learning_rate * db
    
    if epoch%100==0:
        print(loss)

print(W,b)#----I want to delete this too!

######################################################################
# Code to send W and b goes here!
# or, you may consider making the above a function/class
# and calling it from the code that actually sends the variables.
# In such a case, The above function will return W,b
######################################################################

#Prediction on Training set
preds = []
for i in sigmoid(Z):
	if i>0.5:
		preds.append(1)
	else:
		preds.append(0)

print("Score :",f1_score(preds,y_train))

count = 0
for i in range(len(y_train)):
	if y_train[i]==preds[i]:
		count = count + 1

print()
print(count,len(y_train))
print(count/len(y_train))			#Efficiency, wrong meassure to use!
print()

#########################################################################
# In case of IoT in project, The edge device should do the following :	#
# Z = temp * W[0] + RH * W[1] + b 										#
# a = 1/(1+e**(-Z))														#
# if a>0.5:																#
#     "Fire!"															#
# else:																	#
#     "No Fire"															#
#########################################################################

#Prediction on test set##################################################
Z = np.matmul(x_test, W) + b
test_preds = []
for i in sigmoid(Z):
	if i>0.5:
		test_preds.append(1)
	else:
		test_preds.append(0)

print("Score :",f1_score(test_preds,y_test))

count = 0
for i in range(len(y_test)):
	if y_test[i]==test_preds[i]:
		count = count + 1

print(count,len(y_test))
print(count/len(y_test))
W = W.tolist()
W = [W[0][0],W[1][0]]
b = b.tolist()
b = b[0][0]
print(W[0],W[1],b)

while True:
	temp = input('Temperature :')
	RH = input('Relative Humidity :')

	Z = float(temp) * W[0] + float(RH) * W[1] + b
	a = 1/(1+exp(-Z))
	print(a)
	if a>0.5:
		print("Fire!")
	else:
		print("No Fire")

