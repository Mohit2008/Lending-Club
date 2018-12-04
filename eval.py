import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import log_loss
import time
warnings.filterwarnings("ignore")
start = time.time()
np.random.seed(3064) # set the random seed




def logloss(y,p):
    total= np.empty(len(y))
    if len(p)!=len(y):
        raise Exception('Lengths of prediction and labels do not match.')
    if  any(i<0 for i in p):
        raise Exception('Negative probability provided.')
    p=np.maximum(np.minimum(p, 1-10**(-15)), 10**(-15))
    for index,elem in enumerate(p):
        if y[index]==1:
            total[index]=-np.log(elem)
        else:
            total[index]= -np.log(1-elem)
    return np.mean(total)

result = pd.DataFrame(index=['Test-Set1', 'Test-Set2', 'Test-Set3'], columns=["model1", "model2",'model3'], dtype='float64')
for i in range(1,4):
    first_submission='Eval_Data/mysubmission1_'+str(i)+'.txt'
    second_submission='Eval_Data/mysubmission2_'+str(i)+'.txt'
    third_submission='Eval_Data/mysubmission3_'+str(i)+'.txt'
    test_file='Eval_Data/test'+str(i)+'.csv'
    test=pd.read_csv(test_file, header=0, usecols=['id', 'loan_status'])
    first = pd.read_csv(first_submission, header=0)
    second = pd.read_csv(second_submission, header=0)
    third = pd.read_csv(third_submission, header=0)
    inner1 = pd.merge(test, first , how='left', on='id')
    inner2 = pd.merge(test, second , how='left', on='id')
    inner3 = pd.merge(test, third , how='left', on='id')
    update_dict={"loan_status" :{'Fully Paid':0, 'Charged Off':1,'Default':1}}
    inner1.replace(update_dict,inplace=True)
    inner2.replace(update_dict,inplace=True)
    inner3.replace(update_dict,inplace=True)
    out1=log_loss(inner1.loan_status.values, inner1.prob.values)
    out2=log_loss(inner2.loan_status.values, inner2.prob.values)
    out3=log_loss(inner3.loan_status.values, inner3.prob.values)
    result.loc['Test-Set'+str(i),"model1"]= np.round(out1,4)
    result.loc['Test-Set'+str(i),"model2"]= np.round(out2,4)
    result.loc['Test-Set'+str(i),"model3"]= np.round(out3,4)
print(result.mean())
