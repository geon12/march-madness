"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()

#change to stage 2
#comment out less <2015

seasonres = pd.read_csv(direc+'/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

seeds = pd.read_csv(direc+'/MDataFiles_Stage1/MNCAATourneySeeds.csv')

res = pd.read_csv(direc+'/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
confres = pd.read_csv(direc+'/MDataFiles_Stage1/MConferenceTourneyGames.csv')




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

def conftournelo(curr,year):
    k =20
    games = confres[confres['Season']==year]
    for i in range(0,len(games)):
        wteam = games['WTeamID'].iloc[i]
        lteam = games['LTeamID'].iloc[i]
        locw = list(teams).index(wteam)
        locl = list(teams).index(lteam)
        elw = curr[locw]
        ell = curr[locl]
        
        transw = np.power(10, elw/400)
        transl = np.power(10, ell/400)
        
        expw = transw/(transw+transl)
        expl = transl/(transw+transl)
        
        
        curr[locw] = curr[locw] + k*(1-expw)
        curr[locl] = curr[locl] + k*-1*expl
        
    return curr


   #!!!!!!!!!!!!!!!
numb =21 #Change to 22!!!!!!!!!!

teamelo = np.ones((numb+1,len(teams)))*1500
teamw = np.unique(res['WTeamID'])
teaml = np.unique(res['LTeamID'])
for j in range(0,len(teams)):
    if(teams[j] not in teamw):
        teamelo[:,j] = .85* teamelo[:,j]
    if(teams[j] not in teaml):
        teamelo[:,j] = .8* teamelo[:,j]
        
   

     
for i in range(0,numb):
    
    curr= teamelo[i]
    year= i +1998
    
    teamelo[i] = calcelo(curr,year)
    if(year>2000):
        teamelo[i] = conftournelo(curr,year)
    #!!!!!!!coment out
    if(year<2015):#comment out for final
        teamelo[i] = calctournelo(teamelo[i],year)
    
    teamelo[i+1] = .75*teamelo[i] + .25*1502
    
    
    
"""
For stage2
Change date!!!!!!!!!! and numb
"""

teamelo[numb] = calcelo(teamelo[numb],2019) # change date!!!!

massey = pd.read_csv(direc+"/MDataFiles_Stage1/MMasseyOrdinals.csv") #update to latest massey ordinals
    

def calcProb(team1,team2,year):
    loc1 = list(teams).index(team1)
    loc2 = list(teams).index(team2)
    el1 = teamelo[year-1998][loc1]
    el2 = teamelo[year-1998][loc2]

    #may have to change massey dayrankingnum        
    massey2 = massey[['TeamID','OrdinalRank']][massey['SystemName']=='POM'][massey['Season']==year][massey['RankingDayNum']==133]

    rank1 = massey2['OrdinalRank'][massey2['TeamID']==team1].values[0]
    rank2 =  massey2['OrdinalRank'][massey2['TeamID']==team2].values[0]
    el1 = el1+ -700/350*rank1 +(700/350+700)
    el2 = el2+ -700/350*rank2 +(700/350+700)
    
    prob1 = 1/(1 + np.power(10, (el2-el1)/400))
    
    return prob1
    

sub = pd.read_csv('MSampleSubmissionStage1_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]

prob = []
for i in range(0,len(idlist)):
    print(i)
    
    pr = calcProb(int(idlist[i][1]),int(idlist[i][2]),int(idlist[i][0]))
    
    prob.append(pr)
#sub['Pred'] = pd.read_csv('stage2men1.csv')['pred']
#sub['Pred'] = (sub['Pred']+prob)/2.
    
sub['Pred'] = prob

sub.to_csv('stage1men2.csv',header=['ID','Pred'],index=False)



    



