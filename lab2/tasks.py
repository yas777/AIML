import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.1 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(2, 0.1, 100, 50)
	nn1.addLayer(FullyConnectedLayer(2,4))
	nn1.addLayer(FullyConnectedLayer(4,2))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.2 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(2, 0.01, 100, 30)
	nn1.addLayer(FullyConnectedLayer(2,2))
	nn1.addLayer(FullyConnectedLayer(2,2))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(10, 0.01, 100, 30)
	nn1.addLayer(FullyConnectedLayer(784,30))
	nn1.addLayer(FullyConnectedLayer(30,10))

	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	
	XTrain = XTrain[0:5000,:,:,:]
	XVal = XVal[0:1000,:,:,:]
	XTest = XTest[0:1000,:,:,:]
	YVal = YVal[0:1000,:]
	YTest = YTest[0:1000,:]
	YTrain = YTrain[0:5000,:]
	
	modelName = 'model.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	# # TASK 2.4 - YOUR CODE HERE
	nn2 = nn.NeuralNetwork(10, 0.01, 100, 30)
	# nn2.addLayer(ConvolutionLayer([3,32,32], [10,10], 5, 2))
	# nn2.addLayer(ConvolutionLayer([5,12,12], [4,4], 5, 2))
	# nn2.addLayer(FlattenLayer())
	# nn2.addLayer(FullyConnectedLayer(125,10))
	# nn2 = nn.NeuralNetwork(10, 0.01, 100, 30)
	nn2.addLayer(ConvolutionLayer([3,32,32], [10,10], 10, 2))
	nn2.addLayer(AvgPoolingLayer([10,12,12], [4,4], 4))
	nn2.addLayer(FlattenLayer())
	nn2.addLayer(FullyConnectedLayer(90,10))
	# nn2.addLayer(FullyConnectedLayer(100,10))
	
	###################################################
	return nn2,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION


	nn2.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=True, saveModel=True, modelName=modelName)
	pred, acc = nn2.validate(XTest, YTest)
	print('Test Accuracy ',acc)