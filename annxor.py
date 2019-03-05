# the structure of neural network: 
#    input layer with 2 inputs
#    1 hidden layer with 2 units, tanh()
#    output layer with 1 unit, sigmoid()

import numpy as np
import scipy
from scipy.special import expit
import matplotlib.pyplot as plt

def loadData(filename):
  # load data from filename into X
  X=[]
  
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    words = line.split(" ")
    # convert value of first attribute into float
    Row = []
    Row.append(float(1))
    for word in words:
        Row.append(float(word))
    
    X.append(Row)
    
  return np.asarray(X)


# X:(4,4)
# W is a list, W[0] is hidden layer params (2x3), W[1] is output layer params (1x3)
# Returns:
# intermRslt[0]:oh, hidden layer output
# intermRslt[1]:ino, ouput layer input, add row of 1s on top of oh
# intermRslt[2]:oo, output layer output, (1x4), 4 is number of training examples
def feedforward(X, W):
    X_input = X[:, :-1]
    Wh = W[0]
    Oh = np.tanh(np.matmul(Wh,X_input.T))
    Ino = []
    Ino.append([1,1,1,1])
    for i in range(Oh.shape[0]):
        New_row = []
        for j in range(Oh.shape[1]):
            New_row.append(Oh[i,j])
        Ino.append(New_row)
    
    Ino = np.array(Ino)
    Oo = expit(np.matmul(W[1],Ino))
    
    return [Oh, Ino, Oo]
    

def paraIni():
  #code for fixed network and initial values
  
  # parameters for hidden layer, 3 by 3 
  wh=np.random.uniform(low = -1.0, high = 1.0, size=(2,3))
  #wh=np.array([[0.1859,-0.7706,0.6257],[-0.7984,0.5607,0.2109]])
  
  # parameter for output layer 1 by 3
  wo=np.random.uniform(low = -1.0, high = 1.0, size=(1,3))
  #wo=np.array([[0.1328,0.5951,0.3433]])

  return [wh,wo]

# Y (4x1)
# Yhat (1x4)
def errCompute(Y, Yhat):
    m = Y.shape[0]
    J = (1/(2 * m)) * np.sum((Y - Yhat.T)**2)
    return J
    

def backpropagate(X, W, intermRslt, alpha):
    X_input = X[:, :-1]
    Y = X[:,-1]
    Y = Y.reshape((1,4))
    Oo = intermRslt[2]
    d0 = np.multiply(np.multiply((Y - Oo),Oo),(1-Oo))
    
    Ino = intermRslt[1]
    Woutput = W[1]
    Woutput_update = Woutput + alpha * (np.matmul(d0,Ino.T))/4.0
    W[1] =Woutput_update
    
    Woutput_prime = Woutput[:, 1:]
    
    Oh = intermRslt[0]
    temp = 1-np.multiply(Oh,Oh)
    dh = np.multiply(np.matmul(Woutput_prime.T,d0),temp)
    
    Whidden = W[0]
    Whidden_update = Whidden + alpha*(np.matmul(dh,X_input))/4.0
    
    W[0] = Whidden_update
    return W
    

def runTest(filename,numIteration, alpha):
    R = FFMain(filename,numIteration, alpha)
    X = loadData(filename)
    Outputs = feedforward(X, R[2])
    Oo = Outputs[2]
    with open("FeedOutput.txt", 'w') as outfile:
        for i in range(Oo.shape[0]):
            for j in range(Oo.shape[1]):
                outfile.write(str(Oo[i][j]))
                outfile.write('\n')
    
  
def FFMain(filename,numIteration, alpha):
  #data load
  X = loadData(filename)
  #
  W = paraIni()
  
  #number of features
  n = X.shape[1]
  
  #error
  errHistory = np.zeros((numIteration,1))
  
  plt.clf()
  title = str("Alpha:" + str(alpha) + " and NumOfIteration:" + str(numIteration))
  plt.title(title)  
  Error = []
  Iter_list = range(0,numIteration)
  for i in range(numIteration):
    #feedforward
    intermRslt=feedforward(X,W)
    #Cost function
    errHistory[i,0]=errCompute(X[:,n-1:n],intermRslt[2])
    Error.append(float(errHistory[i,0]))
    #backpropagate
    W=backpropagate(X,W,intermRslt,alpha)

  Yhat=np.around(intermRslt[2]) 
  plt.plot(Iter_list,Error)
  name = str("error.png")
  plt.savefig(name, format="png")
  return [errHistory,intermRslt[2],W]