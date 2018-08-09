import numpy as np

# x = (hours slleeping, hours studying), y = score on test
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
x = x/np.amax(x, axis=0) # maximum of x array
y = y/100 # max test score is 100

# w1 = np.random.randn(2,3)
# print(w1)
# z = np.dot(x, w1)
# print(z)

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

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, x, y, o):
        # backword propagate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # applying drivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contributed
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #applying derivative of sigmoid to z2 error

        self.W1 += x.T.dot(self.z2_delta) #adjusting first set (intput --> hiddent) weights
        self.W2 += self.z2.T.dot(self.o_delta) #adjusting second set (hidden --> output) weights

    def train(self, x, y):
        o = self.forword(x)
        self.backward(x, y, o)    

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
    print("Input: \n" + str(x))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.forword))
    print("Loss: \n" + str(np.mean(np.square(y - NN.forword(x))))) # mean sum squared  loss
    print("\n")
    NN.train(x, y)

#defining our output
o = NN.forword(x)
print("Predicted Outpu: \n" + str(o))
print("Actual Output: \n" + str(y))

xPredicted = np.array(([4,8]), dtype=float)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the 