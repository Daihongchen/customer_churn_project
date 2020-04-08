# Customer Churn Project

We selected the customer churn project as our Module 3 team project for two main reasons:
1. This is a business problem that would apply to other workplaces, or as an example for the job interviews. 
2. The data is pretty clean and does not need too much preprocessing work, so that we can focus on the modeling. 

The data is downloaded from kaggle: https://www.kaggle.com/becksddf/churn-in-telecoms-dataset/kernels

## Business Questions:

1. Build a classifier to predict whether a customer will ("soon")stop doing business with SyriaTel, a telecommunications company.
2. Most naturally, your audience here would be the telecom business itself, interested in losing money on customers who don't stick around very long. Are there any predictable patterns here?

## Project goal:

> Successfully predict customers who will stop doing business with SyriaTel, so that the company would not lose money on these customers.

## Process:

1. Preprocessing data:
    - Use HotOneEncoder dummy-coded the 'state' variable.
    - Coded 'international plan', 'voice mail plan' into '1' and '0' from string.
    - Define features X, and target y.
    - Split X_train y_train into subsets: X_train1, X_test1, y_train1, y_test1
    - Scaled X_train1, X_test1 Using StandardScaler
    - The target data is imbalanced, True 15%; False 85%.
    - Create a new feature of 'total charge' that is the sum of all the charges.
2. Modeling:
    - We selected StackingClassifier as our algorithm.
       - Model stacking is also called meta-stacking, where multiple models are stacked, and their predictions are aggregated. In this case, more models are better because the more likely they have the potential to pick up different characteristics of the data.
    - We created two functions:
       - `metrics`: it is a function to generate accuracy, precision, recall, and f-1 scores.
       - `make_model`: it is a function that generate the results from StackingClassifier with four parameters: X_train, X_test, y_train, y_test.
          - Estimators (submodels) selection: We selected K-Nearest-Neighbors(KNN), RandomForest, LogisticRegression, and GradientBoosting as submodels first, but the results showed that KNN generates had very poor performance on prediction with low scores on accuracy, precision, recall, and f-1. Therefore, we removed KNN from the estimators of the StackingClassifier algorithm. We used logistic Regression model as the final model which is the default as well.
    - Model results: we got relatively good model performance using Stacking.
    - Model improvement:
       - The specific goal is to improve the model performance of correctly predicting a customer will stop doing business, which is the TRUE value in the target data. More specifically, we want our model predicts correctly when it is true, and reduce false negative, which means it is predicted false while it is actually true.
       - Therefore, we want a high recall score. Our initial recall score is around 0.75.
    - Feature selections/engineering: In order to increase the recall score, we tried to remove some features that have small values on feature importances. When removing dummyed 'state' variable, the recall score increases. Removing the other features resulted in decreasing the recall scores. Therefore, we keeps all features but 'state'.
3. Model evaluation: We generated the confusion matrix and applied the function named 'metrics' to generate accuracy, precision, recall, and f-1 scores.
    - Final model: StackingClassifier with the following parameters:
      ```
      estimators = [
        ('rf', RandomForestClassifier()),
        ('log', LogisticRegression(solver = 'liblinear')),
        ('grad', GradientBoostingClassifier())
      ], final_estimator = 'LogisticRegression()',
      cv = 5
      ```
    - Model evaluation: Accuracy: 0.98; Precision: 1.0;  Recall: 0.89;  F1: 0.94
4. Pickle: we created a pickle file named model.pickle. Using pickle, we can call the model without re-training.
