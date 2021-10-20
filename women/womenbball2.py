import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()

wstats = np.load('womenstats.npy')
wtest = np.load('womenstatstest.npy')
sub = pd.read_csv('WSampleSubmissionStage1_2020.csv')#change to stage2, change other files

answers =[]
for i in range(0,int(len(wstats)/2)):
    answers.append(1)
    answers.append(0)
    
answers = np.array(answers)
from xgboost import XGBClassifier

xgb = XGBClassifier(seed=11)

xgb.fit(wstats,answers)

prob = xgb.predict_proba(wtest)


sub['Pred'] = prob[:,1]

sub.to_csv('stage1women1.csv',header=['ID','Pred'],index=False)#change to stage 2