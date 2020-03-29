######################################################################
# This module was developed by Harish A and Saksham Jhawar
# This module is a part of Improving Iot edge computing
# Further modules have to be added accordingly.
# Recent build 20th April 2019
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

X=[]
Y=[]

#Loading fire positive data###########################################
with open('fireFinal.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for row in csv_reader:
		if not row[0]=='temp':
			inputList.append([float(row[0]),float(row[1]),1])
			outputList.append(1)

X=[]
Y=[]

#Loading fire negative data###########################################
itr = 0
with open('noFire.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for row in csv_reader:
		if not row[0]=='temp':
			inputList.append([float(row[0])*5,float(row[1])*100,0])
			outputList.append(0)
		if itr>600:
			break
		itr = itr + 1

print(type(inputList))

with open('fire.csv','w') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerows(inputList)