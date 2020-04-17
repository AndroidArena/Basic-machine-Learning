# Boston House Price Prediction Project on UCI dataset
I have tried to predict Boston House price from - https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
Steps I performed - 
1. Convert the data into a readable format, done some cleaning
2. Understand the problem objective
3. Decide whether its a regression or classification problem
4. import necessary libraries and read the data into DataFrame
5. Done some exploratory Data Analysis (EDA)
6. Spilt the Data between Training and Test sets
6. Done Stratified shuffle for proper distribution of data points in test and train set like 0/1 values of column CHAS
6. find some co-relation between attributes and understand that which feature is important for us.
7.Assign data to X_train , y_train
8.Done Preprocessing step using sklearn pipeline- Imputer-median to fill NA values and Standardization for feature scalling
9. Apply a model one by one and understand its performance -
a. LinearRegression
b. Decision Tree
c. Random forest 
10. evaluate Model using RMSE and Furthur Cross-validation. 
11. Finally, choose the Random forest model to further predict test_set
12. Save model  using joblib

For Any suggestion kindly update your .py within this branch.
- Thank You

