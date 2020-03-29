
from math import exp
w0 = -0.022
w1 = -0.097
b = 5.79

for i in range(0,100,5):
	for j in range(0,100,5):
		z = w0*i+w1*j+b
		a = 1/(1+exp(-z))
		if a<0.5:
			print(i,j)
		j=j+4
	i=i+4