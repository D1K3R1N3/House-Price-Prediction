# Set the URL of the dataset
url <- "https://github.com/obakis/econ_data/raw/master/hprice2.rds"

# Download the dataset
download.file(url, destfile = "hprice2.rds", mode = "wb")

# Load the dataset
data <- readRDS("hprice2.rds")

# Load required packages
library(neuralnet)
library(randomForest)
library(arm)
library(class)
library(e1071)
library(xgboost)
library(rpart)

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(nrow(data), nrow(data) * 0.8)  # 80% for training
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Convert the data to DMatrix format
dtrain <- xgb.DMatrix(as.matrix(train_data[, -1]), label = train_data$price)
dtest <- xgb.DMatrix(as.matrix(test_data[, -1]), label = test_data$price)


# Random Forest
rf_formula <- as.formula("price ~ crime + nox + rooms + dist + radial + proptax + stratio + lowstat + lprice + lnox + lproptax")
rf_model <- randomForest(rf_formula, data = train_data, ntree = 100)  # Use 100 trees
rf_predicted_prices <- predict(rf_model, newdata = test_data)

# Bayesian Model
bayes_formula <- as.formula("price ~ crime + nox + rooms + dist + radial + proptax + stratio + lowstat + lprice + lnox + lproptax")
bayes_model <- bayesglm(bayes_formula, data = train_data, family = gaussian())
bayes_predicted_prices <- predict(bayes_model, newdata = test_data)

# K-Nearest Neighbors (KNN)
knn_formula <- as.formula("price ~ crime + nox + rooms + dist + radial + proptax + stratio + lowstat + lprice + lnox + lproptax")
knn_model <- knn(train = train_data[, -1], test = test_data[, -1], cl = train_data$price, k = 5)
knn_predicted_prices <- as.numeric(levels(knn_model))[knn_model]

# Decision Tree
tree_formula <- as.formula("price ~ crime + nox + rooms + dist + radial + proptax + stratio + lowstat + lprice + lnox + lproptax")
tree_model <- rpart(tree_formula, data = train_data)
tree_predictions <- predict(tree_model, newdata = test_data)

# Support Vector Regression (SVR)
svr_formula <- as.formula("price ~ crime + nox + rooms + dist + radial + proptax + stratio + lowstat + lprice + lnox + lproptax")
svr_model <- svm(svr_formula, data = train_data)
svr_predictions <- predict(svr_model, newdata = test_data)

# Set the parameters for XGBoost
params <- list(
  objective = "reg:squarederror",  # Regression objective
  eval_metric = "rmse",  # Evaluation metric: root mean squared error
  max_depth = 6,  # Maximum tree depth
  eta = 0.1,  # Learning rate
  nrounds = 100  # Number of boosting rounds
)
# Train the XGBoost model
xgb_model <- xgboost(params, data = dtrain, nrounds = params$nrounds)

# Make predictions on the test data
xgb_predictions <- predict(xgb_model, dtest)





# Calculate MSE for each model
rf_mse <- mean((test_data$price - rf_predicted_prices)^2)
bayes_mse <- mean((test_data$price - bayes_predicted_prices)^2)
knn_mse <- mean((test_data$price - knn_predicted_prices)^2)
svr_mse <- mean((test_data$price - svr_predictions)^2)
xgb_mse <- mean((test_data$price - xgb_predictions)^2)
tree_mse <- mean((test_data$price - tree_predictions)^2)

# Calculate R-squared for each model
rf_r_squared <- 1 - sum((test_data$price - rf_predicted_prices)^2) / sum((test_data$price - mean(test_data$price))^2)
bayes_r_squared <- 1 - sum((test_data$price - bayes_predicted_prices)^2) / sum((test_data$price - mean(test_data$price))^2)
knn_r_squared <- 1 - sum((test_data$price - knn_predicted_prices)^2) / sum((test_data$price - mean(test_data$price))^2)
svr_r_squared <- 1 - sum((test_data$price - svr_predictions)^2) / sum((test_data$price - mean(test_data$price))^2)
xgb_r_squared <- 1 - sum((test_data$price - xgb_predictions)^2) / sum((test_data$price - mean(test_data$price))^2)
tree_r_squared <- 1 - sum((test_data$price - tree_predictions)^2) / sum((test_data$price - mean(test_data$price))^2)


cat("Random Forest MSE:", rf_mse, "\n")
cat("Bayesian Model MSE:", bayes_mse, "\n")
cat("KNN MSE:", knn_mse, "\n")
cat("Support Vector Regression MSE:", svr_mse, "\n")
cat("XGBoost MSE:", xgb_mse, "\n")
cat("Decision Tree MSE:", tree_mse, "\n")

cat("Random Forest R-squared:", rf_r_squared, "\n")
cat("Bayesian Model R-squared:", bayes_r_squared, "\n")
cat("KNN R-squared:", knn_r_squared, "\n")
cat("Support Vector Regression R-squared:", svr_r_squared, "\n")
cat("XGBoost R-squared:", xgb_r_squared, "\n")
cat("Decision Tree R-squared:", tree_r_squared, "\n")
