import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import time
warnings.filterwarnings("ignore")
start = time.time()
np.random.seed(3064) # set the random seed

def get_variable_by_type(loans_train):
    categorical = [var for var in loans_train.columns if loans_train[var].dtype=='O'] # get all categorical variable
    numerical = [var for var in loans_train.columns if loans_train[var].dtype!='O'] # get all numerical variable
    categorical.remove("loan_status") # remove the target variable
    discrete = []
    continous=[]
    for var in numerical:
            if len(loans_train[var].unique())<15:
                discrete.append(var) # identify all discrete columns
            else:
                continous.append(var) # identify all continous columns
    return categorical, discrete, continous, numerical


def impute_missing_values(loans_train,loans_test, numerical,categorical):
    loans_train.fillna(loans_train[numerical].mean(), inplace=True) # fill numerical data with mean as a replacement of missing value
    loans_test.fillna(loans_train[numerical].mean(), inplace=True)
    for col in categorical:
        loans_train[col].fillna(loans_train[col].mode()[0], inplace=True) # fill categorical data with mode as a replacement of missing value
        loans_test[col].fillna(loans_train[col].mode()[0], inplace=True)
    return loans_train, loans_test

# fix the issue of outliers
def clip_outliers(train_df,variables):
    try:
        for var in variables:
            IQR = train_df[[var]].quantile(0.75) - train_df[[var]].quantile(0.25) # get the interquantile range
            Lower_fence = float(train_df[[var]].quantile(0.25) - (IQR * 3)) # set lower bound
            Upper_fence = float(train_df[[var]].quantile(0.75) + (IQR * 3)) # set upper bound
            train_df[var]=train_df[var].clip(Lower_fence, Upper_fence) # clip boundaries
        return train_df
    except Exception as ex:
        print("Error occured in clipping outliers in data due to {}".format(ex))
        raise ex

def do_feature_processing(loans_train, loans_test,categorical, discrete):
    loans_train['term'] = loans_train['term'].apply(lambda s: np.int8(s.split()[0])) # extract the first part of the term in train
    loans_test['term'] = loans_test['term'].apply(lambda s: np.int8(s.split()[0])) # extract the first part of the term in test
    categorical.remove('term') # update the categorical record
    discrete.append('term') # update the dicrete record
    loans_train['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True) # replace string in train
    loans_test['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True) # replace string in test

    loans_train['emp_length'].replace('< 1 year', '0 years', inplace=True) # replace string in train
    loans_test['emp_length'].replace('< 1 year', '0 years', inplace=True)# replace string in test

    loans_train['emp_length'] = loans_train['emp_length'].apply(lambda s: np.int8(s.split()[0])) # extract the inetger portion of emp length
    loans_test['emp_length'] = loans_test['emp_length'].apply(lambda s: np.int8(s.split()[0]))
    categorical.remove('emp_length') # update the categorical record
    discrete.append('emp_length') # update the dicrete record

    update_dict={"grade" : {"G" : 0, "F" : 1, "E" : 2, "D": 3, "C" : 4, "B" : 5, 'A':6},
            "sub_grade" : {"G5" : 0, "G4" : 1, "G3" : 2, "G2": 3, "G1" : 4,
                          "F5" : 5, "F4" : 6, "F3" : 7, "F2": 8, "F1" : 9,
                          "E5" : 10, "E4" : 11, "E3" : 12, "E2": 13, "E1" : 14,
                          "D5" : 15, "D4" : 16, "D3" : 17, "D2": 18, "D1" : 19,
                          "C5" : 20, "C4" : 21, "C3" : 22, "C2": 23, "C1" : 24,
                          "B5" : 25, "B4" : 26, "B3" : 27, "B2": 28, "B1" : 29,
                          "A5" : 30, "A4" : 31, "A3" : 32, "A2": 33, "A1" : 34 } }
    loans_train.replace(update_dict,inplace=True) # integer encoding in train
    loans_test.replace(update_dict,inplace=True) # integer encoding in test
    categorical.remove('grade') # update the categorical record
    discrete.append('grade') # update the dicrete record
    categorical.remove('sub_grade') # update the categorical record
    discrete.append('sub_grade')# update the dicrete record

    loans_train.drop(['title','zip_code','emp_title'], axis=1, inplace=True) # drop non useful columns from train
    loans_test.drop(['title','zip_code','emp_title'], axis=1, inplace=True) # drop non useful columns from test
    categorical.remove('title') # update the categorical record
    categorical.remove('zip_code') # update the categorical record
    categorical.remove('emp_title') # update the categorical record

    loans_train['earliest_cr_line'] = pd.to_datetime(loans_train['earliest_cr_line']) # convert to datetime
    loans_test['earliest_cr_line'] = pd.to_datetime(loans_test['earliest_cr_line'])
    loans_train['year']=loans_train.earliest_cr_line.dt.year # extract year
    loans_train['month']=loans_train.earliest_cr_line.dt.month # extract month
    loans_test['year']=loans_test.earliest_cr_line.dt.year
    loans_test['month']=loans_test.earliest_cr_line.dt.month
    loans_train.drop(['earliest_cr_line'], axis=1, inplace=True) # drop the variable
    loans_test.drop(['earliest_cr_line'], axis=1, inplace=True)
    categorical.remove('earliest_cr_line') # update the categorical record
    discrete.append('year') # update the dicrete record
    discrete.append('month') # update the dicrete record
    return loans_train, loans_test,categorical, discrete

def get_dummies(loans_train, loans_test):
    train_objs_num = len(loans_train)
    trainY= loans_train[["loan_status"]]
    dataset = pd.concat(objs=[loans_train.drop(["loan_status"], axis=1), loans_test], axis=0)
    dataset_preprocessed = pd.get_dummies(dataset) # create dummy for categorical var
    train_df = dataset_preprocessed.iloc[:train_objs_num]
    test_df = dataset_preprocessed.iloc[train_objs_num:]
    return trainY, train_df, test_df


def build_model1(dtrain, dtest,test_df):
    xgb_params = {
    'eta': 0.27,
    'max_depth': 7,
    'subsample': 1.0,
    'colsample_bytree': 0.5,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': 1
    }

    model=xgb.train(xgb_params, dtrain,num_boost_round=70) # train xgbost with above params
    y_test_xgb = model.predict(dtest) # predict on test data
    pred_test_original= pd.DataFrame(np.round(y_test_xgb,2), index=test_df.index) # load prediction in data frame
    pred_test_original.columns=['prob'] # set the column name
    pred_test_original.to_csv(out_file1_path, header=True, index=True, sep=',') # save to disk
    print("Prediction from model 1 persisted to disk as mysubmission1.txt")


def build_model2(train_df, trainY,test_df_scaled,test_df):
    clf = RandomForestClassifier(n_estimators=80, max_depth=18,n_jobs=-1) # train random forest classifier
    clf.fit(train_df.copy(), trainY.copy())
    rfc_pred=clf.predict_proba(test_df_scaled)# predict on test data
    pred_test_original= pd.DataFrame(np.round(rfc_pred[:,1],2), index=test_df.index)# load prediction in data frame
    pred_test_original.columns=['prob']# set the column name
    pred_test_original.to_csv(out_file2_path, header=True, index=True, sep=',') # save to disk
    print("Prediction from model 2 persisted to disk as mysubmission2.txt")

def build_model3(train_df, trainY,test_df_scaled,test_df):
    clf = LogisticRegression(n_jobs=-1).fit(train_df, trainY) # train logistic regression
    logi_pred=clf.predict_proba(test_df_scaled) # predict on test data
    pred_test_original= pd.DataFrame(np.round(logi_pred[:,1],2), index=test_df.index) # load prediction in data frame
    pred_test_original.columns=['prob']# set the column name
    pred_test_original.to_csv(out_file3_path, header=True, index=True, sep=',')# save to disk
    print("Prediction from model 3 persisted to disk as mysubmission3.txt")



#######input config #######
input_train_path='train.csv'
input_test_path='test.csv'
out_file1_path ="mysubmission1.txt"
out_file2_path ="mysubmission2.txt"
out_file3_path ="mysubmission3.txt"


try:
    loans_train=pd.read_csv(input_train_path, index_col=["id"]) # read train data
    loans_test=pd.read_csv(input_test_path, index_col=["id"]) # read test data
    loans_train['loan_status'] = np.where(loans_train['loan_status'] == 'Default', 'Charged Off', loans_train['loan_status']) # replace the loan status calasses
    
    
    categorical, discrete, continous, numerical=get_variable_by_type(loans_train.copy(deep=True)) # get datatype of each var
    loans_train, loans_test=impute_missing_values(loans_train,loans_test, numerical,categorical) # fix the missing values
    loans_train, loans_test,categorical, discrete=do_feature_processing(loans_train, loans_test,categorical, discrete) # perform transformations
    trainY, train_df, test_df=get_dummies(loans_train, loans_test) # create dummy var
    
    train_df.index = train_df.index.astype(np.int64) # set the index to be integer
    test_df.index = test_df.index.astype(np.int64)  # set the index to be integer
    del loans_train # delete unused dataframe
    del loans_test # delete unused dataframe
    train_df=clip_outliers(train_df, continous) # clip the extreme values
    
    
    update_dict={"loan_status" :{'Fully Paid':0, 'Charged Off':1}} # target encoding dic
    trainY.replace(update_dict,inplace=True) # encode target
    sc_X = StandardScaler() # create a scaler
    train_df_scaled = sc_X.fit_transform(train_df) # scale train data
    test_df_scaled = sc_X.transform(test_df) # scale test data
    dtrain = xgb.DMatrix(train_df_scaled, trainY) # load the train and target into Dmatrix for faster operations
    dtest = xgb.DMatrix(test_df_scaled) # load the test into Dmatrix for faster operations
    
    build_model1(dtrain, dtest,test_df) # create model 1
    build_model2(train_df_scaled, trainY,test_df_scaled,test_df) # create model 2
    build_model3(train_df_scaled, trainY,test_df_scaled,test_df) # create model 3
    
    end = time.time() # record time
    print("Script executed successfully in {} seconds ".format(end - start))
except Exception as ex:
    print("Script aborted due to {}".format(ex))