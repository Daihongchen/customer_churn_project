import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_name = '../../data/Customer Churn Data.csv'

def pipeline(pickle = True):
    X_train, _, y_train, _ = get_train_and_test_data()
    model = make_model(X_train, y_train)
    if pickle:
        pickler(model, 'model.pickle')
    return model

    
def get_train_and_test_data():
    '''
    Returns testing and training data
    '''
    data = get_data()
    X_train, X_test, y_train, y_test = split_data(data)
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    return X_train, X_test, y_train, y_test
    
    
def get_data():
    '''
    Gets data from datafile
    
    Returns
    -------
    Returns the data frame to be used in making the model
    '''
    return pd.read_csv(file_name)


def preprocess_data(df):
    """
    Preprocessing steps from baseline model, plus additional steps for final
    model

    Given a dataframe:
    call baseline preprocessing function,
    compute a hybrid feature "total charge",
    drop "state" feature
    """
    # helper function from baseline model
    df = preprocess_data_baseline(df)

    # combine all charges into "total charge" feature
    charge_features = [
        'total day charge',
        'total eve charge',
        'total intl charge',
        'total night charge'
        ]
    df['total charge'] = df[charge_features].sum(axis=1)

    # return df without charge features or "state"
    return df.drop(charge_features + ['state'], axis=1)


def preprocess_data_baseline(df):
    """
    Preprocessing steps for the baseline model, except for one-hot encoding

    Given a raw dataframe:
    convert "international plan" and "voice mail plan" to numeric values,
    drop unnecessary columns
    """
    # convert yes/no to numeric
    df['international plan'] = (df['international plan'] == 'yes').astype(int)
    df['voice mail plan'] = (df['voice mail plan'] == 'yes').astype(int)
    
    # drop unnecessary features
    unused_features = [
        'area code',
        'phone number'
    ]
    df = df.drop(unused_features, axis=1)
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
