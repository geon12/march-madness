"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()

#change to stage2

seasonres = pd.read_csv(direc+'/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')

seeds = pd.read_csv(direc+'/MDataFiles_Stage2/MNCAATourneySeeds.csv')

res = pd.read_csv(direc+'/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
confres = pd.read_csv(direc+'/MDataFiles_Stage2/MConferenceTourneyGames.csv')
detail = pd.read_csv(direc+'/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')




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
numb =22 #Change to 22!!!!!!!!!!

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
    # if(year<2015):#comment out for final , uncomment for stage 1
    #     teamelo[i] = calctournelo(teamelo[i],year)
    
    teamelo[i+1] = .75*teamelo[i] + .25*1502
    
    
    
"""
For stage2
Change date!!!!!!!!!! and numb
"""

teamelo[numb] = calcelo(teamelo[numb],2020) # change date!!!!

massey = pd.read_csv(direc+"/MDataFiles_Stage1/MMasseyOrdinals.csv") #update to latest massey ordinals
    

def rankelo():
    teamelo2 = teamelo.copy()
   
    #change date!!!
    for year in range(1998,2020+1):#change to 2020!!!!!
        massey2 = massey[['TeamID','OrdinalRank']][massey['SystemName']=='POM'][massey['Season']==year][massey['RankingDayNum']==133]
        
        for j in range(0,len(teams)):
            el1 = teamelo[year-1998][j]
            
            if teams[j] in list(massey2['TeamID']):
                
                rank1 = massey2['OrdinalRank'][massey2['TeamID']==teams[j]].values[0]
                el1 = el1+ -700/350*rank1 +(700/350+700)
                teamelo2[year-1998][j] = el1
        
        
    return teamelo2


teamelo_ranked = rankelo()



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
    elw = teamelo_ranked[games1['Season'].loc[i]-1998][locw]
    ell = teamelo_ranked[games1['Season'].loc[i]-1998][locl]
    game_elos.append(elw-ell)
    
for i in range(0,len(games1)):
    locw = list(teams).index(games1['WTeamID'].loc[i])
    locl = list(teams).index(games1['LTeamID'].loc[i])
    elw = teamelo_ranked[games1['Season'].loc[i]-1998][locw]
    ell = teamelo_ranked[games1['Season'].loc[i]-1998][locl]
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
    """
    added in 2021/Possible alternative for points per possession calculation
    From https://thepowerrank.com/cbb-analytics/
    POSS = FGA – OREB + TO + (0.475 * FTA)
    
    comment out other teamA['ppp']
    
    teamA['twor'] =teamA['winloss']*teamA['WOR'] + (1-teamA['winloss'])*teamA['LOR'] #offensive rebounds
    team['ppp'] =teamA['score']/(teamA['tfga'] -team['twor']+ teamA['twto']+ .475*teamA['tfta'])
    """
    return (teamA['ppp'].median(),teamA['ppp'].std())



"""
Remember not to include 2015 training data
"""
"""
Comment out this limiting time
"""

sub = pd.read_csv('MSampleSubmissionStage2_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]


import scipy.stats

prob = []

for i in range(0,len(idlist)):
    print(i)
    
    locw = list(teams).index(int(idlist[i][1]))
    locl = list(teams).index(int(idlist[i][2]))
    elw = teamelo_ranked[int(idlist[i][0])-1998][locw]
    ell = teamelo_ranked[int(idlist[i][0])-1998][locl]
    elodiff = np.array([elw - ell]).reshape(1,-1)
    movpred = linear.predict(elodiff)
    
    
    ppm1, ppstd1 = teamstats(int(idlist[i][0]),int(idlist[i][1]))
    ppm2, ppstd2 = teamstats(int(idlist[i][0]),int(idlist[i][2]))  
    mean21 = ppm2-ppm1
    ppstd = np.sqrt(ppstd1*ppstd1 + ppstd2*ppstd2)
    pr = scipy.stats.norm(mean21, ppstd).cdf(movpred[0][0]/70.0)
    prob.append(pr)
    
sub['Pred'] = prob

sub.to_csv('stage2men2.csv',header=['ID','Pred'],index=False)

    
    
    



    
    
