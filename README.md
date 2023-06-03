House Price Prediction Models

This project aims to predict house prices using various machine learning algorithms in the R programming language. The dataset used for training and testing the models can be downloaded from the following URL: Dataset URL.

To run this project, please follow these steps:

Set the URL of the dataset by assigning it to the 'url' variable in the code.

Download the dataset using the 'download.file()' function. The dataset will be saved as "hprice2.rds" in the current working directory.

Load the required packages by executing the 'library()' function for each package used in the code: neuralnet, randomForest, arm, class, e1071, xgboost, and rpart.

Split the data into training and testing sets using the 'sample()' function to randomly select 80% of the data for training. Adjust the seed value ('set.seed()') for reproducibility.

Convert the data to the required format for each algorithm. For XGBoost, the 'xgb.DMatrix()' function is used to create the training and testing matrices.

Train and test the following models:

Random Forest: Uses the 'randomForest()' function with the desired formula and parameters.
Bayesian Model: Uses the 'bayesglm()' function from the 'arm' package with the desired formula and family.
K-Nearest Neighbors (KNN): Uses the 'knn()' function with the training data, testing data, and desired value for 'k'.
Decision Tree: Uses the 'rpart()' function with the desired formula and data.
Support Vector Regression (SVR): Uses the 'svm()' function with the desired formula and data.
XGBoost: Sets the parameters in the 'params' list and uses the 'xgboost()' function with the training data.
Calculate the mean squared error (MSE) and R-squared for each model using the predictions and the actual house prices.

The results will be displayed in the console, showing the MSE and R-squared values for each model.

Please note the following:

This project assumes that you have an active internet connection to download the dataset.
Adjust the formulas, parameters, and settings according to your specific requirements.
The performance of the models is evaluated using MSE and R-squared metrics.
For any further questions or assistance, please reach out to ademhat10@gmail.com
