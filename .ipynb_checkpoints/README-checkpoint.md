# customer_churn_project

We selected the Customer churn project as our Modlue 3 team project for two main reasons:
1. This is a business problem that would apply to other workplaces, or as an example for the job interviews. 
2. The data is pretty clean and does not need too much preprocessing work, so that we can focus on the modeling. 

The data is downloaded from kaggle: https://www.kaggle.com/becksddf/churn-in-telecoms-dataset/kernels

Business Questions:

    1. Build a classifier to predict whether a customer will ("soon") stop doing business with SyriaTel, a telecommunications 
    company.
    2. . Most naturally, your audience here would be the telecom business itself, interested in losing money on customers who don't 
    stick around very long. Are there any predictable patterns here?

Project goal: 

    Successfully predict customers who will stop doing business with SyriaTel, so that the company would not lose 
    money on these customers.

Process: 

    1. Preprocessing data: 
        a. Use HotOneEncoder dummy-coded the 'state' varible.
        b. Coded 'international plan', 'voice mail plan' into '1' and '0' from string. 
        3. Define features X, and target y.
        4. Split the X and y to X_train, X_hold, y_train, y_hold. X_hold, y_hold will be used to evaluate the final model. 
        5. Split X_train y_train into subsets: X_train1, X_test1, y_train1, y_test1. 
        6. Scaled X_train1, X_test1 Using  StandardScaler 
        7. The target data is inbalanced, True : False is 
    
    2. Modeling: 
        a. We selected StakingClassifier as our algorithm. 
                Model stacking is also called meta-stacking, where multiple models are stacked, and their predictions are aggregated. In this case, 
                more models are better because the more likely they have the potential to pick up different characteristics of the data.
        b. We created two functions: 
            metircs: it is a funciton to generate accuracy, precision, recall, and f-1 scores. 
            model: it is a funciton that generate the results from StackingClassifier with four parameters: X_train, X_test, y_train, y_test. 
            Estimators (submodels) selection: We selected K-Nearest-Neighbors(KNN), RandomForest, LogisticRegression, and GradientBoosting as submodels first, 
            but the results showed that KNN generates had very poor performance on prediction with low scores on accurcy, precision, recall, and f-1. Therefore,
            we removed KNN from the estimators of the StackingClassifier algorithm. We used logistic Regression model as the final model which is the default as well.
        c. Model results: we got reletively good model performance using Stacking. 
        d. Model improvement: 
            The specific goal is to imporve the model performance of correctly predicting a customer will stop doing business, which is the TRUE 
            value in the target data. More specifically, we want our model predicts correctly when it is true, and reduce false negative, which means 
            it is predicted false while it is actually true. 
            Therefore, we want a high recall score. Our initial recall score is around 0.75. 
        d. feature selections/engineering: In order to increase the recall score, we tried to remove some features that have small values on feature importances. When removing
        dummyed 'state' variable, the recall score increases. Removing the other features resulted in decreasing the recall scores. Therefore, we keeps all features but 'state'.
        
     3. Model evaluation: We generated the confusion matrix and applies the function named 'metrics' to generate accruacy, precision, recall, and f-1 scores. 
          Final model: StackingClassifier with estimators = [('rf', RandomForestClassifier()),('log', LogisticRegression(solver = 'liblinear')), ('grad',       
                                                              GradientBoostingClassifier())] , 
                                          final_estimator = 'LogisticRegression()',
                                                       cv = 5
          Model evaluation: Accuracy: 0.958; Precision: 0.949;  Recall: 0.773;  F1: 0.852
 
 Pickle: 
 
     we create a pickle file named model.pickle. Use pickle, we can call the model. 
     
 



