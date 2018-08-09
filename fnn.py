import numpy as np

# x = (hours slleeping, hours studying), y = score on test
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
x = x/np.amax(x, axis=0) # maximum of x array
y = y/100 # max test score is 100

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix form input to hidden layer # 两行三列
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix form hidden to output layer # 一行三列

    def forword(self, x):
        #forward propapation through our network
        self.z = np.dot(x, self.W1) # dot product of x (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 wieghts
        o = self.sigmoid(self.z3) # final activation fucntion
        return o  

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

NN = Neural_Network()

#defining our output
o = NN.forword(x)
print("Predicted Outpu: \n" + str(o))
print("Actual Output: \n" + str(y))
