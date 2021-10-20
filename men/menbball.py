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
    #!!!!!!!coment out
    if(year<2015):#comment out for final
        teamelo[i] = calctournelo(teamelo[i],year)
    
    teamelo[i+1] = .75*teamelo[i] + .25*1502
    
    
    
"""
For stage2
Change date!!!!!!!!!! and numb
"""

teamelo[numb] = calcelo(teamelo[numb],2019) # change date!!!!



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
    telo = teamelo[yearnum-1998][np.where(teams==teamnum)[0][0]]
    tfga = teamA['tfga'].median()
    tfgma = teamA['tfgma'].median()
    tfta  = teamA['tfta'].median()
    tftma  = teamA['tftma'].median()
    tfgma3 = teamA['tfgma3'].median()
    twor = teamA['twor'].median()
    twdr  = teamA['twdr'].median()
    twto  = teamA['twto'].median()
    
    
    teamB = np.array([wintotal,losstotal, mov_avg,teamseed,telo,tfga,tfgma,tfta,tftma,tfgma3,twor,twdr,twto])
    
    
    return teamB

"""
Remember not to include 2015 training data
"""
"""
Comment out this limiting time
"""
res = res[res['Season']<2015]#change to 2020
#!!!!!!!!!!


res = res[res['Season']>2002]#no detailed results before 2003
res = res.reset_index()

winorlose = []
stats = []
for i in range(0,len(res)):
    print(i)
    stat = np.append(teamstats(res['Season'].loc[i],res['WTeamID'].loc[i]),teamstats(res['Season'].loc[i],res['LTeamID'].loc[i]))
    stats.append(stat)
    winorlose.append(1)
    stat = np.append(teamstats(res['Season'].loc[i],res['LTeamID'].loc[i]),teamstats(res['Season'].loc[i],res['WTeamID'].loc[i]))
    stats.append(stat)
    winorlose.append(0)
    
stats = np.array(stats).astype(float)
winorlose = np.array(winorlose)#just alternating 1 and 0

sub = pd.read_csv('MSampleSubmissionStage1_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]

np.save('menstats',stats)

stats2 = []
for i in range(0,len(idlist)):
    print(i)
    stat = np.append(teamstats(int(idlist[i][0]),int(idlist[i][1])),teamstats(int(idlist[i][0]),int(idlist[i][2])))
    stats2.append(stat)
    
stats2 = np.array(stats2).astype(float)

np.save('menstatstest',stats2)