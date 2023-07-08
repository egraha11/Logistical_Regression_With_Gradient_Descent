import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification




class Logistical_Regression:

    def __init__(self):

        #learning rate for gradient descent
        self.learning_rate = .1
        self.threshold = .5
        self.b0 = np.random.rand()
        self.b1 = np.random.rand()       


    def fit(self, x, y):
        for radius in range(len(x)):

            #create odds using logistical regression intercept and probability
            odds = self.b0 + self.b1*x[radius]

            #use the sigmoid function to create a probability
            probability = 1/(1 + np.exp(-(odds)))

            #use binary cross entropy for a loss function
            loss = (y[radius] * probability) + ((1 - y[radius]) * (1 - probability))

            #calculate gradient with respect to b1 and update coefficient
            self.b1 -= ((probability - y[radius]) * x[radius]) * self.learning_rate

            #calculate gradient with respect to b0 and update intercept
            self.b0 -= (probability - y[radius]) * self.learning_rate

        sigmoid_points = np.arange(np.min(x), np.max(x), .01)

        y_points = [1/(1 + np.exp(-(self.b0 + self.b1*point))) for point in sigmoid_points]

        plt.scatter(x, y)
        plt.plot(sigmoid_points, y_points)
        plt.show()


    def predict(self, x):

        preds = np.empty([len(x), 2])

        for sample in range(len(x)):

            preds[sample][0] = 1/ (1 + np.exp(-(self.b0 + self.b1*x[sample])))

            if preds[sample][0] > .5:
                preds[sample][1] = 1
            else:
                preds[sample][1] = 0

        return preds



def main():

    data = make_classification(n_samples=2000, n_features=4, n_classes=2)

    x = np.array(data[0])
    x = x[:, 0]
    y = np.array(data[1])


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)

    model = Logistical_Regression()

    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    print(accuracy_score(y_test, preds[:, 1]))


if __name__ == '__main__':
    main()



