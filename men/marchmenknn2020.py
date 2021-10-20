"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

import operator
import math

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
    

def rankelo():
    teamelo2 = teamelo.copy()
   
    for year in range(1998,2019+1):#change to 2020!!!!!
        massey2 = massey[['TeamID','OrdinalRank']][massey['SystemName']=='POM'][massey['Season']==year][massey['RankingDayNum']==133]
        
        for j in range(0,len(teams)):
            el1 = teamelo[year-1998][j]
            
            if teams[j] in list(massey2['TeamID']):
                
                rank1 = massey2['OrdinalRank'][massey2['TeamID']==teams[j]].values[0]
                el1 = el1+ -700/350*rank1 +(700/350+700)
                teamelo2[year-1998][j] = el1
        
        
    return teamelo2


teamelo_ranked = rankelo()


games1 = seasonres[['WTeamID','LTeamID','Season']][seasonres['Season']>=2003]
games2 = confres[['WTeamID','LTeamID','Season']][confres['Season']>=2003]
games1 = pd.concat([games1, games2], ignore_index=True)

def knn(ids):
    yearnum2 = 0
    games = 0
    prob = []
    for i in range(0,len(idlist)):
        team1 = int(idlist[i][1])
        team2 = int(idlist[i][2])
        yearnum = int(idlist[i][0])
    
        if yearnum != yearnum2:
            yearnum2 = yearnum
            games = games1[games1['Season']<=yearnum]
            games3 = res[['WTeamID','LTeamID','Season']][confres['Season']>=2003]
            games3 = games3[games3['Season']<yearnum]
            games = pd.concat([games, games3], ignore_index=True)
        pr = calc_prob(games,team1, team2,yearnum)
        prob.append(pr)
    
    return prob
        
        
def calc_prob(games,team1, team2,yearnum):
    teamelo1 = teamelo_ranked[yearnum-1998][np.where(teams==team1)[0][0]]
    teamelo2 = teamelo_ranked[yearnum-1998][np.where(teams==team2)[0][0]]
    
    num = 100
    
    dists = []
    for i in range(0,len(games)):
        dist = 0
        gm = games.loc[i]
        eloW = teamelo_ranked[gm['Season']-1998][np.where(teams==gm['WTeamID'])[0][0]]
        eloL = teamelo_ranked[gm['Season']-1998][np.where(teams==gm["LTeamID"])[0][0]]
        dist1 = math.sqrt(pow(teamelo1-eloW,2)+pow(teamelo2-eloL,2))
        dist2 = math.sqrt(pow(teamelo2-eloW,2)+pow(teamelo1-eloL,2))
        if dist1 <= dist2:
            win = 1
            dist = dist1
        else:
            win = 0
            dist =dist2
        
        dists.append([win, dist])
    dists.sort(key=operator.itemgetter(1))

    wins = np.array(dists)[:,0]                                          
    pr = wins.sum()/num
    return pr


 

sub = pd.read_csv('MSampleSubmissionStage1_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]

prob= knn(idlist[0:3])
    
#sub['Pred'] = prob

#sub.to_csv('stage1men2.csv',header=['ID','Pred'],index=False)



    



