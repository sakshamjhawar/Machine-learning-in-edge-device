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

X1=[]
Y1=[]
X2=[]
Y2=[]
#Loading fire positive data###########################################
with open('weatherHistory.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for row in csv_reader:
		if not row[0]=='Formatted Date':
			temp = float(row[3])
			RH = float(row[5])*100
			if row[1]=='Overcast' and len(X1)<1000  and temp>0 and float(row[5])*100<=170*(25-float(row[3]))/23:
				inputList.append([temp,RH])
				outputList.append(1)

				X1.append(temp)
				Y1.append(RH)

			elif row[1]=='Clear' and len(X2)<1000 and float(row[5])*100>130*(25-float(row[3]))/23>0:
				inputList.append([temp,RH])
				outputList.append(0)

				X2.append(temp)
				Y2.append(RH)


#plt.scatter(X1, Y1, color="r")
#plt.ylabel('Relative Humidity')
#plt.xlabel('Temperature')
#plt.scatter(X2, Y2, color="g")

X = np.array(inputList)
y = np.array(outputList)
plt.show()

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
        print("Loss after",epoch,"iterations is",loss)

print()
print("Variables W and b are ",W,b)#----I want to delete this too!

plt.scatter(x_train[:,0], x_train[:, 1], c=y_train.ravel())
ax = plt.gca()
xvals = np.array(ax.get_xlim()).reshape(-1,1)
yvals = -(xvals * W[0][0] + b)/ W[1][0]

plt.plot(xvals,yvals)
plt.ylim(0,100)

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

print(x_test.shape,W.shape)
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
Z = float(temp) * W[1] + float(RH) * W[0] + b
a = 1/(1+exp(-Z))
print(a)

Z = float(temp) * W[0] + float(RH) * W[1] + b
a = 1/(1+exp(-Z))
print(a)

if a<0.5:
	print("Fire!")
else:
	print("No Fire")

plt.scatter([float(temp)],[float(RH)],color='g')
plt.show()