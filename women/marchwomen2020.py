"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()


seasonres = pd.read_csv(direc+'/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

seeds = pd.read_csv(direc+'/WDataFiles_Stage1/WNCAATourneySeeds.csv')

res = pd.read_csv(direc+'/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')





teams = np.unique(seasonres['WTeamID'])

def calcelo(curr,year):
    k =20
    games = seasonres[seasonres['Season']==year]
    for i in range(0,len(games)):
        wteam = games['WTeamID'].iloc[i]
        lteam = games['LTeamID'].iloc[i]
        locw = list(teams).index(wteam)
        locl = list(teams).index(lteam)
        elw = curr[locw]
        ell = curr[locl]
        if( games['WLoc'].iloc[i]=='H'):
            elw = elw +150
        if( games['WLoc'].iloc[i]=='A'):
            ell = ell +150
        
        transw = np.power(10, elw/400)
        transl = np.power(10, ell/400)
        
        expw = transw/(transw+transl)
        expl = transl/(transw+transl)
        
        
        curr[locw] = curr[locw] + k*(1-expw)
        curr[locl] = curr[locl] + k*-1*expl
        
    return curr

def calctournelo(curr,year):
    k =20
    games = res[res['Season']==year]
    for i in range(0,len(games)):
        wteam = games['WTeamID'].iloc[i]
        lteam = games['LTeamID'].iloc[i]
        locw = list(teams).index(wteam)
        locl = list(teams).index(lteam)
        elw = curr[locw]
        ell = curr[locl]
        if( games['WLoc'].iloc[i]=='H'):
            elw = elw +150
        if( games['WLoc'].iloc[i]=='A'):
            ell = ell +150
        
        transw = np.power(10, elw/400)
        transl = np.power(10, ell/400)
        
        expw = transw/(transw+transl)
        expl = transl/(transw+transl)
        
        
        curr[locw] = curr[locw] + k*(1-expw)
        curr[locl] = curr[locl] + k*-1*expl
        
    return curr
   
"""        
change numb to 22 in stage
"""
#!!!!!!!!!!!!!!!
numb =21

teamelo = np.ones((numb+1,len(teams)))*1500
teamw = np.unique(res['WTeamID'])
teaml = np.unique(res['LTeamID'])
for j in range(0,len(teams)):
    if(teams[j] not in teamw):
        teamelo[:,j] = .85* teamelo[:,j]
    if(teams[j] not in teaml):
        teamelo[:,j] = .8* teamelo[:,j]
        
"""        
#Change 20 in loop to 21 for stage 2
"""        
for i in range(0,numb):
    
    curr= teamelo[i]
    year= i +1998
    
    teamelo[i] = calcelo(curr,year)
    if(year<2015):#comment out for final
        teamelo[i] = calctournelo(teamelo[i],year)
    
    teamelo[i+1] = .75*teamelo[i] + .25*1502
    
    
    
"""
change dates for stage 2, change 2018 to 2019
"""

teamelo[numb] = calcelo(teamelo[numb],2019) #change date



def calcProb(team1,team2,year):
    loc1 = list(teams).index(team1)
    loc2 = list(teams).index(team2)
    el1 = teamelo[year-1998][loc1]
    el2 = teamelo[year-1998][loc2]
    seed1 =seeds['Seed'][seeds['TeamID']==team1][seeds['Season']==year].values[0].strip('WXYZab')
    seed1 = int(seed1)
    seed2 =seeds['Seed'][seeds['TeamID']==team2][seeds['Season']==year].values[0].strip('WXYZab')
    seed2 = int(seed2)
    
    
    
    el1 = el1 + 1700/30*(seed1-seed2)
        
    prob1 = 1/(1 + np.power(10, (el2-el1)/400))
    
    return prob1
    

sub = pd.read_csv('WSampleSubmissionStage1_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]



prob = []
for i in range(0,len(idlist)):
    print(i)
    pr = calcProb(int(idlist[i][2]),int(idlist[i][1]),int(idlist[i][0]))
    
    prob.append(pr)
    
sub['Pred'] = prob


#sub['Pred'] = pd.read_csv('stage1women1.csv')['pred']#change to stage 2
#sub['Pred'] = (sub['Pred']+prob)/2.

sub.to_csv('stage1women2.csv',header=['ID','Pred'],index=False)#change to stage 2



    



