import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        trainingData = np.asarray(trainingData)
        n = trainingData.shape[0]
        for i in range(self.num_classifiers):
            # print(i)
            ind = np.random.choice(n,int(self.ratio*n))
            td = np.take(trainingData,ind)
            tl = np.take(trainingLabels,ind)
            self.classifiers[i].train(td,tl)            

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """
        guesses = np.zeros(len(data))
        for i in range(len(self.classifiers)):
            # print(self.classifiers[i].classify(data))
            guesses = guesses + np.asarray(self.classifiers[i].classify(data))
        for j in range(len(guesses)):
            if (guesses[j] > 0):
                guesses[j] = 1
            if (guesses[j] == 0):
                guesses[j] = np.random.choice([-1,1])
            if (guesses[j] < 0):
                guesses[j] = -1;
        return guesses.tolist()
        
