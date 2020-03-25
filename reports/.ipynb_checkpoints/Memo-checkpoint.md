## Customer Chur Classification Project Memo

Title: 

    Predicting Customer Churn in SyrialTel Company using StackingClassifier Algorithm

Business problem: 

    SyrialTel company is losing money on customers who stop doing business with them after a shortwhile of service. For example, customer singed for a promotion as a new user, but churn after the promotion is over. By predicitng if a customer will churn or not, SyrialTel could allocate more resources on the ones that will keep doing business with the company. 

Objective: 

    This project is aimed to build a sophisticated machine learning algorithm that help SyrialTel company successfully predict if a customer will soon stop doing business with the company based on historical data. So that the company will not lose money on these who will not do business with the company soon. 
    
Methods and Model Selection:

    In order to optimizing the classification between customers who will soon stop doing business with SyrialTel and who will keep doing business, we selecetd StackingClassifier algorithm, where multiple models are stacked, and their predictions are aggregated. In this case, more models are better because the more likely they have the potential to pick up different characteristics of the data.
    
    We initially stacked K-nearest-neighbors, randorm forest, logistic regression, and gradient boosting as submodels or estimators. KNN was removed because of its poor performance based on our evaluation metrics: accuracy, precision, recall, and f-1 scores. 

Final model: 

          StackingClassifier with estimators = [('rf', RandomForestClassifier()),
                                                ('log', LogisticRegression(solver = 'liblinear')),                                                     
                                                ('grad', GradientBoostingClassifier())] , 
                             final_estimator = 'LogisticRegression()',
                                          cv = 5
          Model evaluation: Accuracy: 0.958; Precision: 0.949;  Recall: 0.773;  F1: 0.852


Business implication:

    Our model has a relatively high performance according to the evaluation metrics which include accuracy, precision, recall, and F-1 score. We are confident to implement our model in the business. A caution is that the recall score is 0.773, which means the model has 23% chance to misclassify these who would stop doing business with SyriaTel as will continue doing business. 


Limitations of the model: 

    The target variable in the dataset is inbalanced. There are only 15% of target variable are True, and 85% are False. The purpose of the modeling is to correctly detect the customers who with a True value, and reduce the chance when it fails to detect the True values, which means when it is positive, but was calssfied as negative, the false negative. Therefore, we want a high recall score in our model.

    With a few feature engineering and selections, we have the final model with a recall as 0.77. It is satisfactory, but not ideal. We want to inform this score to the leadership team of the company that, we have 23% chance missclassifying the customers who will stop doing business as customers who would continue doing the business with the comany. 


Unexpected results during the experiments:

    Unexpected signs of coefficients, after we dummuy code the state, the model performance decreased on the test dataset. This might because a categorical predictor variable can have only a small number of observations having a certain value. In a classification problem, these few observations by chance may have the same response value. Such “perfect” predictors may cause overfitting, especially when tree-based models are used. One conservative approach is to remove all dummy variables corresponding to rare categories. A cutoff of 10 observations is recommended (Luo,& e.t, 2016). Based on this recommendation and the difference of model performance, we removed the feature of 'state'



References:

Luo, W., Phung, D., Tran, T., Gupta, S., Rana, S., Karmakar, C., Shilton, A., Yearwood, J., Dimitrova, N., Ho, T. B., Venkatesh, S., & Berk, M. (2016). Guidelines for Developing and Reporting Machine Learning Predictive Models in Biomedical Research: A Multidisciplinary View. Journal of medical Internet research, 18(12), e323. https://doi.org/10.2196/jmir.5870
