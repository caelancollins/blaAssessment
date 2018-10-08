# Skills Assessment


In this repository:

* predict.py: Python code using the scikit-learn machine learning library to predict values on given data

* PredictiveModelingAssessmentData.csv: given data used to test and build model

* TestData.csv: given data to use model on

* TestDataPredictions.csv: given data with model predicted 'y' values

* lotteryProbabilties.csv: complete table of conditional probabilities of hypothetical new lottery system

* lottery.py: python code used to calculate lottery probabilities

### Prediction Process
This prediction model was built using scikit-learn's Support Vector Machine with a RBF Kernel. This model was to found to produce much more accurate results than other linear based regression models.
Although preprocessing is generally recommended before SVM model building, I could not find methods to improve the predicability of the model. After deciding on the SVR and RBF kernel, parameters were then explored to greater reduce the models error, although the defaults provided the most sensible parameters.
