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
import random

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

X1=[]
Y1=[]
X2=[]
Y2=[]

#Loading fire positive data###########################################
with open('Rain.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	i = 1
	for row in csv_reader:
		if i==1 or row[5]=='NA' or row[4]=='NA' or row[3]=='NA' or row[1]=='NA':
			i=i+1
			continue
		i=i+1
		if i==1500:
			break
		RH = float(row[5])*100
		temp= (float(row[4])+float(row[3]))/2
		if row[1]=='Mostly Cloudy' or row[1]=='Overcast':
			X1.append(temp)
			Y1.append(RH)
			inputList.append([temp,RH])
			outputList.append(1)
		else:
			RH = RH/2
			X2.append(temp)
			Y2.append(RH)
			inputList.append([temp,RH])
			outputList.append(0)

plt.scatter(X1, Y1, color="g")
plt.ylabel('Relative Humidity')
plt.xlabel('Temperature')
plt.scatter(X2, Y2, color="r")


X = np.array(inputList)
y = np.array(outputList)

#Randomly merging data################################################
perm = np.random.permutation(1497)

#Dividing data into test and train####################################
x_train, x_test = X[perm][300:], X[perm][:300]
y_train, y_test = y[perm][300:], y[perm][:300]
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
        print("Loss after",epoch,"iterations is",loss)

print()
print("Variables W and b are ",W[0],W[1],b)#----I want to delete this too!

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

print()
print("Training set score :",f1_score(preds,y_train))

count = 0
for i in range(len(y_train)):
	if y_train[i]==preds[i]:
		count = count + 1

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

print("Test set Score :",f1_score(test_preds,y_test))

count = 0
for i in range(len(y_test)):
	if y_test[i]==test_preds[i]:
		count = count + 1
		#if y_test[i]==1:
		#	print(x_test[i])

print(count,len(y_test))
print(count/len(y_test))

temp = input('Temperature :')
RH = input('Relative Humidity :')

W = W.tolist()
W = [W[0][0],W[1][0]]
b = b.tolist()
b = b[0][0]

print(temp,W[0],RH,W[1],b)
Z = float(temp) * W[0] + float(RH) * W[1] + b
a = 1/(1+exp(-Z))
print(a)

plt.show()