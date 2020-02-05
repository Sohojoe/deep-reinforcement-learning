import numpy as py
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


w1 = 2
w2 = 6
b = -2  
output = w1*0.4 + w2*0.6 + b
prob = sigmoid(output)
print (output, prob)

w1 = 3
w2 = 5
b = -2.2
output = w1*0.4 + w2*0.6 + b
prob = sigmoid(output)
print (output, prob)

w1 = 5
w2 = 4
b = -3
output = w1*0.4 + w2*0.6 + b
prob = sigmoid(output)
print (output, prob)

print("- - ")
for n in [-100,-100,-10,-1,-.1-.01,-.001,0,.001,.01,.1,1,10,100,1000]:
    print (n, sigmoid(n))
    


print("- - ")
w1=-1
w2=-1
x1=1
x2=1
output = w1*x1+w2*x2+b
prob = sigmoid(output)
print (output, prob)
x1=10
x2=10
output = w1*x1+w2*x2+b
prob = sigmoid(output)
print (output, prob)