"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()

#change to stage2

seasonres = pd.read_csv(direc+'/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

seeds = pd.read_csv(direc+'/MDataFiles_Stage1/MNCAATourneySeeds.csv')

res = pd.read_csv(direc+'/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
confres = pd.read_csv(direc+'/MDataFiles_Stage1/MConferenceTourneyGames.csv')
detail = pd.read_csv(direc+'/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')




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
    #!!!!!!!coment out , uncomment for stage 1
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



games1 = seasonres[['WTeamID','LTeamID','Season','WScore', 'LScore']][seasonres['Season']>=2003]
games2 = res[['WTeamID','LTeamID','Season','WScore', 'LScore']][res['Season']>=2003]
#Comment out for final!!!!!
games2 = games2[games2['Season']<2015]#!!!!Comment out for final
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

linear = linear_model.RANSACRegressor()
#linear = linear_model.LinearRegression()
linear.fit(np.asarray(game_elos).reshape(-1,1),np.array(scorediff1).reshape(-1,1))


def teamstats(yearnum,teamnum):
    teamA = detail[detail['Season']==yearnum]
    a = teamA[teamA['WTeamID']==teamnum]
    b = teamA[teamA['LTeamID']==teamnum]
    teamA = pd.concat([a,b]).reset_index()
    #teamA['DayNum'] = (teamA['DayNum']-detail['DayNum'][detail['Season']==yearnum].min())/(detail['DayNum'][detail['Season']==yearnum].max()-detail['DayNum'][detail['Season']==yearnum].min())
    teamA['winloss'] = 0
    teamA['winloss'][teamA['WTeamID']==teamnum]=1
    
    teamA['mov'] = 0#margin of victory
    teamA['mov'] = (2*teamA['winloss']-1)*teamA['WScore'] + (1-2*teamA['winloss'])*teamA['LScore']
#    teamA['opp_elo'] = 0
#    
#    teamA['opp_elo'] = teamA['winloss']*teamA['LTeamID'] + (1-teamA['winloss'])*teamA['WTeamID']
#
#    vals = teamA['opp_elo'].values
#    for i in range(0,len(vals)):
#        #loca = np.where(teams==vals[i])[0][0]
#        loca = list(teams).index(vals[i])
#        teamA['opp_elo'].loc[i] = teamelo[yearnum-1998][loca]
        
    
    #team stats    
    teamA['tfga'] = teamA['winloss']*teamA['WFGA'] + (1-teamA['winloss'])*teamA['LFGA']
    teamA['tfgma'] = teamA['winloss']*teamA['WFGM']/teamA['WFGA'] + (1-teamA['winloss'])*teamA['LFGM']/teamA['LFGA']    
    teamA['tfta'] = teamA['winloss']*teamA['WFTA'] + (1-teamA['winloss'])*teamA['LFTA']
    teamA['tftma'] = teamA['winloss']*teamA['WFTM']/teamA['WFTA'] + (1-teamA['winloss'])*teamA['LFTM']/teamA['LFTA']    
    teamA['tfgma3'] = teamA['winloss']*teamA['WFGM3']/teamA['WFGA3'] + (1-teamA['winloss'])*teamA['LFGM3']/teamA['LFGA3']    
    teamA['twor'] = teamA['winloss']*teamA['WOR'] + (1-teamA['winloss'])*teamA['LOR']
    teamA['twdr'] = teamA['winloss']*teamA['WDR'] + (1-teamA['winloss'])*teamA['LDR']
    teamA['twto'] = teamA['winloss']*teamA['WTO'] + (1-teamA['winloss'])*teamA['LTO']
    teamA['tfgafg3'] =  (teamA['winloss']*teamA['WFGA3'] + (1-teamA['winloss'])*teamA['LFGA3'] ) / teamA['tfga']
   
    teamA['score'] = teamA['winloss']*teamA['WScore'] + (1-teamA['winloss'])*teamA['LScore']
    
    teamA['ppp'] = teamA['score']/(teamA['tfga'] + teamA['twto']+teamA['tfta']) #points per possesion
    #teamA = teamA[['winloss','mov']]
    
    #teamB = teamA.nlargest(15,'opp_elo')#pick 15 most important games
    
    #teamB = teamB.values.flatten()
    
    wintotal = teamA['winloss'].sum()
    losstotal = teamA.shape[0] - wintotal
    mov_avg = teamA['mov'].mean()
    if teamnum in seeds['TeamID'][seeds['Season']==yearnum].values:
        teamseed = seeds['Seed'][seeds['TeamID']==teamnum][seeds['Season']==yearnum].values[0].strip('WXYZab')
    else:
        teamseed =18
    telo = teamelo_ranked[yearnum-1998][np.where(teams==teamnum)[0][0]]
    tfga = teamA['tfga'].median()
    tfgma = teamA['tfgma'].median()
    tfta  = teamA['tfta'].median()
    tftma  = teamA['tftma'].median()
    tfgma3 = teamA['tfgma3'].median()
    twor = teamA['twor'].median()
    twdr  = teamA['twdr'].median()
    twto  = teamA['twto'].median()
    tfgafg3 = teamA['tfgafg3'].median()
    ppp = teamA['ppp'].median()
    
    
    teamB = np.array([wintotal,losstotal, mov_avg,teamseed,telo,tfga,tfgma,tfta,tftma,tfgma3,twor,twdr,twto,tfgafg3,ppp])
    
    
    return teamB

"""
Remember not to include 2015 training data
"""
"""
Comment out this limiting time
"""
#Uncomment for stage 1

res = res[res['Season']<2015]#comment out for final,uncomment for stage 1

#!!!!!!!!!!


res = res[res['Season']>2002]#no detailed results before 2003
res = res.reset_index()

winorlose = []
stats = []
for i in range(0,len(res)):
    print(i)
    stat = np.append(teamstats(res['Season'].loc[i],res['WTeamID'].loc[i]),teamstats(res['Season'].loc[i],res['LTeamID'].loc[i]))
    locw = list(teams).index(res['WTeamID'].loc[i])
    locl = list(teams).index(res['LTeamID'].loc[i])
    elw = teamelo_ranked[games1['Season'].loc[i]-1998][locw]
    ell = teamelo_ranked[games1['Season'].loc[i]-1998][locl]
    elodiff = np.array([elw - ell]).reshape(1,-1)
    movpred = linear.predict(elodiff)
    stat = np.append(stat,movpred)
    stats.append(stat)
    winorlose.append(1)
    stat = np.append(teamstats(res['Season'].loc[i],res['LTeamID'].loc[i]),teamstats(res['Season'].loc[i],res['WTeamID'].loc[i]))
    elodiff = np.array([ell - elw]).reshape(1,-1)
    movpred = linear.predict(elodiff)
    stat = np.append(stat,movpred)
    stats.append(stat)
    winorlose.append(0)
    
stats = np.array(stats).astype(float)
winorlose = np.array(winorlose)#just alternating 1 and 0

sub = pd.read_csv('MSampleSubmissionStage1_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]

#np.save('menstats',stats)

stats2 = []
for i in range(0,len(idlist)):
    print(i)
    stat = np.append(teamstats(int(idlist[i][0]),int(idlist[i][1])),teamstats(int(idlist[i][0]),int(idlist[i][2])))
    locw = list(teams).index(int(idlist[i][1]))
    locl = list(teams).index(int(idlist[i][2]))
    elw = teamelo_ranked[int(idlist[i][0])-1998][locw]
    ell = teamelo_ranked[int(idlist[i][0])-1998][locl]
    elodiff = np.array([elw - ell]).reshape(1,-1)
    movpred = linear.predict(elodiff)
    stat = np.append(stat,movpred)
    stats2.append(stat)
    
stats2 = np.array(stats2).astype(float)

#np.save('menstatstest',stats2)

#Then go to menbball2.py



answers =[]
for i in range(0,int(len(stats)/2)):
    answers.append(1)
    answers.append(0)
    
answers = np.array(answers)
from xgboost import XGBClassifier

xgb = XGBClassifier(seed=11)

xgb.fit(stats,answers)

prob = xgb.predict_proba(stats2)


sub['Pred'] = prob[:,1]

sub.to_csv('stage1men1.csv',header=['ID','Pred'],index=False)



    
    
