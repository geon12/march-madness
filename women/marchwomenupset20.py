"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()

#change to stage 2
seasonres = pd.read_csv(direc+'/WDataFiles_Stage2/WRegularSeasonCompactResults.csv')

seeds = pd.read_csv(direc+'/WDataFiles_Stage2/WNCAATourneySeeds.csv')

res = pd.read_csv(direc+'/WDataFiles_Stage2/WNCAATourneyCompactResults.csv')
detail = pd.read_csv(direc+'/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv')




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
numb =22

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
    # if(year<2015):#comment out for final
    #     teamelo[i] = calctournelo(teamelo[i],year)
    
    teamelo[i+1] = .75*teamelo[i] + .25*1502
    
    
    
"""
change dates for stage 2, change 2018 to 2019
"""

teamelo[numb] = calcelo(teamelo[numb],2020) #change date



games1 = seasonres[['WTeamID','LTeamID','Season','WScore', 'LScore']][seasonres['Season']>=2003]
games2 = res[['WTeamID','LTeamID','Season','WScore', 'LScore']][res['Season']>=2003]
#Comment out for final!!!!!
#games2 = games2[games2['Season']<2015]#!!!!Comment out for final
games1 = pd.concat([games1, games2], ignore_index=True)

scorediff1 =  games1['WScore'] - games1['LScore']
scorediff2 = games1['LScore'] - games1['WScore']
scorediff1 = pd.concat([scorediff1, scorediff2], ignore_index=True)


game_elos = []


for i in range(0,len(games1)):
    locw = list(teams).index(games1['WTeamID'].loc[i])
    locl = list(teams).index(games1['LTeamID'].loc[i])
    if len(seeds['Seed'][seeds['TeamID']==games1['WTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]]) == 1 :
        seed1 =seeds['Seed'][seeds['TeamID']==games1['WTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]].values[0].strip('WXYZab')
        seed1 = int(seed1)
    else:
        seed1 = 18
        
    if len(seeds['Seed'][seeds['TeamID']==games1['LTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]]) == 1 :
        seed2 =seeds['Seed'][seeds['TeamID']==games1['LTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]].values[0].strip('WXYZab')
        seed2 = int(seed2)
    else:
        seed2 = 18
        
    elw = teamelo[games1['Season'].loc[i]-1998][locw]
    ell = teamelo[games1['Season'].loc[i]-1998][locl]
    
    elw = elw - 1700/30*(seed1-seed2)
    game_elos.append(elw-ell)
    
for i in range(0,len(games1)):
    locw = list(teams).index(games1['WTeamID'].loc[i])
    locl = list(teams).index(games1['LTeamID'].loc[i])
    
    
    
    if len(seeds['Seed'][seeds['TeamID']==games1['WTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]]) == 1 :
        seed1 =seeds['Seed'][seeds['TeamID']==games1['WTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]].values[0].strip('WXYZab')
        seed1 = int(seed1)
    else:
        seed1 = 18
        
    if len(seeds['Seed'][seeds['TeamID']==games1['LTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]]) == 1 :
        seed2 =seeds['Seed'][seeds['TeamID']==games1['LTeamID'].loc[i]][seeds['Season']==games1['Season'].loc[i]].values[0].strip('WXYZab')
        seed2 = int(seed2)
    else:
        seed2 = 18
    
    
    
    
    
    elw = elw - 1700/30*(seed1-seed2)
    
    elw = teamelo[games1['Season'].loc[i]-1998][locw]
    
    ell = teamelo[games1['Season'].loc[i]-1998][locl]
    game_elos.append(ell-elw)
    
from sklearn import linear_model


"Predicts margin of victory from elo rating difference"

#linear = linear_model.RANSACRegressor()
linear = linear_model.LinearRegression()
linear.fit(np.asarray(game_elos).reshape(-1,1),np.array(scorediff1).reshape(-1,1))


def teamstats(yearnum,teamnum):
    teamA = detail[detail['Season']==yearnum]
    a = teamA[teamA['WTeamID']==teamnum]
    b = teamA[teamA['LTeamID']==teamnum]
    teamA = pd.concat([a,b]).reset_index()
    #teamA['DayNum'] = (teamA['DayNum']-detail['DayNum'][detail['Season']==yearnum].min())/(detail['DayNum'][detail['Season']==yearnum].max()-detail['DayNum'][detail['Season']==yearnum].min())
    teamA['winloss'] = 0
    teamA['winloss'][teamA['WTeamID']==teamnum]=1
        
    
    #team stats    
    teamA['tfga'] = teamA['winloss']*teamA['WFGA'] + (1-teamA['winloss'])*teamA['LFGA']
    
    teamA['twto'] = teamA['winloss']*teamA['WTO'] + (1-teamA['winloss'])*teamA['LTO']
    teamA['tfta'] = teamA['winloss']*teamA['WFTA'] + (1-teamA['winloss'])*teamA['LFTA']
    teamA['score'] = teamA['winloss']*teamA['WScore'] + (1-teamA['winloss'])*teamA['LScore']
    
    teamA['ppp'] = teamA['score']/(teamA['tfga'] + teamA['twto']+teamA['tfta']) #points per possesion
    
    return (teamA['ppp'].median(),teamA['ppp'].std())

sub = pd.read_csv('WSampleSubmissionStage2_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]


import scipy.stats

prob = []

for i in range(0,len(idlist)):
    print(i)
    
    locw = list(teams).index(int(idlist[i][1]))
    locl = list(teams).index(int(idlist[i][2]))
    elw = teamelo[int(idlist[i][0])-1998][locw]
    ell = teamelo[int(idlist[i][0])-1998][locl]
    seed1 =seeds['Seed'][seeds['TeamID']==int(idlist[i][1])][seeds['Season']==int(idlist[i][0])].values[0].strip('WXYZab')
    seed1 = int(seed1)
    seed2 =seeds['Seed'][seeds['TeamID']==int(idlist[i][2])][seeds['Season']==int(idlist[i][0])].values[0].strip('WXYZab')
    seed2 = int(seed2)
    
    
    
    elw = elw - 1700/30*(seed1-seed2)
    elodiff = np.array([elw - ell]).reshape(1,-1)
    movpred = linear.predict(elodiff)
    
    
    ppm1, ppstd1 = teamstats(int(idlist[i][0]),int(idlist[i][1]))
    ppm2, ppstd2 = teamstats(int(idlist[i][0]),int(idlist[i][2]))  
    mean21 = ppm2-ppm1
    ppstd = np.sqrt(ppstd1*ppstd1 + ppstd2*ppstd2)
    pr = scipy.stats.norm(mean21, ppstd).cdf(movpred[0][0]/75.0)
    prob.append(pr)
    
sub['Pred'] = prob


sub.to_csv('stage2women2.csv',header=['ID','Pred'],index=False)#change to stage 2



    



