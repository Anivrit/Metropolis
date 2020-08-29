import struct, gzip
import numpy as np

global train_x
global test_x
global train_y_one_hot
global test_y_one_hot

def loadY(fnlabel):
	f = gzip.open(fnlabel, 'rb')
	f.read(8)
	return np.frombuffer(f.read(), dtype = np.uint8)

def loadX(fnimg):
	f = gzip.open(fnimg, 'rb')
	f.read(16)
	return np.frombuffer(f.read(), dtype = np.uint8).reshape((-1, 28*28))


trainX = loadX("train-images-idx3-ubyte.gz")
trainY = loadY("train-labels-idx1-ubyte.gz")
testX = loadX("t10k-images-idx3-ubyte.gz")
testY = loadY("t10k-labels-idx1-ubyte.gz")
data = {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY}

def process(data):
    transposed = np.transpose(data)
    return (transposed.astype('double')/256)

train_x = process(data["trainX"])
test_x = process(data["testX"])
train_y_one_hot = np.zeros((10,60000))
for i in range(60000):
    for j in range(10):
        train_y_one_hot[j][i] = (data["trainY"][i]==j)
test_y_one_hot = np.zeros((10,10000))
for k in range(10000):
    for l in range(10):
        test_y_one_hot[l][k] = (data["testY"][k]==l)
