import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()

mstats = np.load('menstats.npy')
mtest = np.load('menstatstest.npy')
sub = pd.read_csv('MSampleSubmissionStage1_2020.csv')#change to stage2, change other files

answers =[]
for i in range(0,int(len(mstats)/2)):
    answers.append(1)
    answers.append(0)
    
answers = np.array(answers)
from xgboost import XGBClassifier

xgb = XGBClassifier(seed=11)

xgb.fit(mstats,answers)

prob = xgb.predict_proba(mtest)


sub['Pred'] = prob[:,1]

sub.to_csv('stage1men1.csv',header=['ID','Pred'],index=False)
