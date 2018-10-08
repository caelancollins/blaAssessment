import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
import math
import os

trainFileName = '~/Downloads/PredictiveModelingAssessmentData.csv'
testFileName = '~/Downloads/TestData.csv'
predictionFileName = 'TestDataPredictions.csv'

dfTrain = pd.read_csv(trainFileName)
dfTest = pd.read_csv(testFileName)

X = dfTrain[['x1', 'x2']]
y = dfTrain[['y']]

m = svm.SVR(kernel = 'rbf')
m.fit(X,y.values.ravel())

dfTest['y'] = m.predict(dfTest[['x1','x2']])

dfTest.to_csv(predictionFileName,index=False)







#os.system('say "The program is complete"')
