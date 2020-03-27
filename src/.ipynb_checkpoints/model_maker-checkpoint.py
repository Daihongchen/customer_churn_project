import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

file_name = '../../data/Customer Churn Data.csv'

def pipeline(pickle = True):
    X_train, X_test, y_train, y_test = get_train_and_test_data()
    model = make_model(X_train, y_train)
    if pickle:
        pickler(model, 'model.pickle')
    return model

    
def get_train_and_test_data():
    '''
    Returns testing and training data
    '''
    data = get_data()
    return split_data(data)
    
    
def get_data():
    '''
    Gets data from datafile and does some pruning.
    Drops columns that worsen the model and agregates the charges columns (This helps the model)
    
    Returns
    -------
    Returns the data frame to be used in making the model
    '''
    df = pd.read_csv(file_name)
    
    df['international plan'] = (df['international plan'] == 'yes').astype(int)
    df['voice mail plan'] = (df['voice mail plan'] == 'yes').astype(int)

    df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total intl charge'] + df['total night charge']
    df = df.drop(['total day charge', 'total eve charge', 'total intl charge', 'total night charge'], axis = 1)
    
    df = df.drop(['area code', 'phone number', 'state'], axis = 1)
    return df
    
    
def split_data(data):
    '''
    Does a train test split on the passed in with churn as the target
    
    Parameters
    ----------
    data: churn data to be split
    
    Returns
    -------
    Training predictors, test predictor, training target, test target
    '''
    target = data['churn']
    X = data.copy()
    X = X.drop(['churn'], axis = 1)
    return train_test_split(X, target, test_size = 0.30, random_state = 42)


def make_model(X_train, y_train):
    '''
    fits and returns a stacking model based on the data passed in
    '''
    estimators = [('rf', RandomForestClassifier()),
                  ('log', LogisticRegression(solver = 'liblinear')),
                  ('grad', GradientBoostingClassifier())]
    stack = StackingClassifier(estimators = estimators, final_estimator = LogisticRegression(), cv = 5)
    stack.fit(X_train, y_train)
    return stack    
    

def metrics(y_true, y_pred):
    '''
    returns some metrics
    '''
    metric_dictionary = {}
    metric_dictionary['Accuracy'] = str(accuracy_score(y_true, y_pred))
    metric_dictionary['Precision'] = str(precision_score(y_true, y_pred))
    metric_dictionary['Recall'] = str(recall_score(y_true, y_pred))
    metric_dictionary['F1'] = str(f1_score(y_true, y_pred))
    metric_dictionary['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metric_dictionary    
    
    
def pickler(model, file_name):
    '''
    turns a model into a pickle file
    '''
    output_file = open(file_name, 'wb')
    pickle.dump(model, output_file)
    output_file.close()

    
def read_pickle(file_name):
    '''
    reads a pickle file
    '''
    model_file = open(file_name, "rb")
    model = pickle.load(model_file)
    model_file.close()
    return model