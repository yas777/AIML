import util
import numpy as np
import sys
import random
import math 

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        trainingData = np.asarray(trainingData)
        n = trainingData.shape[0]
        sample_weights = np.ones(n)/n
        for i in range(len(self.classifiers)):
            # print("--------" + str(i) + "-----------------")
            classy = self.classifiers[i]
            classy.train(trainingData, trainingLabels, sample_weights.tolist())
            labels = classy.classify(trainingData)
            labels = np.asarray(labels)
            temp_lab = np.asarray(trainingLabels)
            np.place(temp_lab,temp_lab==-1,[0])
            np.place(labels,labels==-1,[0])
            err_cof = np.absolute(temp_lab - labels)
            err = np.sum(sample_weights*err_cof)
            print(err)
            self.alphas[i] = (0.5)*(math.log((1-err)/err))
            wei_cof = err_cof
            np.place(wei_cof,err_cof==0,[-1])
            sample_weights = sample_weights*np.exp(wei_cof*self.alphas[i])
            sample_weights = sample_weights/np.sum(sample_weights)

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
            guesses = guesses + self.alphas[i]*np.asarray(self.classifiers[i].classify(data))
        for j in range(len(guesses)):
            if (guesses[j] > 0):
                guesses[j] = 1
            if (guesses[j] == 0):
                guesses[j] = np.random.choice(self.legalLabels)
            if (guesses[j] < 0):
                guesses[j] = -1;
        return guesses.tolist()