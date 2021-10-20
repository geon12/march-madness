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


#!!!!!!!!!!!!!!!change to 22 for stage 2
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
#Change 21 in loop to 22 for stage 2
remove if less than 2015!!!!!!
"""

for i in range(0,numb):
    
    curr= teamelo[i]
    year= i +1998
    
    teamelo[i] = calcelo(curr,year)
    #if(year<2015):#comment out for final
    #!!!!Remove if less than 2014 for stage 2!!!!!
        #teamelo[i] = calctournelo(teamelo[i],year)
    
    teamelo[i+1] = .75*teamelo[i] + .25*1502
    
    
    
"""
change dates for stage 2
change both 20s in bracket to 21s 
change 2018 to 2019
"""

teamelo[numb] = calcelo(teamelo[numb],2020)#change to 2019




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
    telo = teamelo[yearnum-1998][np.where(teams==teamnum)[0][0]]
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
Remember not to include 2014 training data
"""
"""
Comment out this limiting time
"""
#res = res[res['Season']<2015]#comment out for 2020

res = res[res['Season']>2009]#no detailed results before 2010
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

sub = pd.read_csv('WSampleSubmissionStage2_2020.csv')#change to stage2, change other files

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]

#np.save('womenstats',stats)

stats2 = []
for i in range(0,len(idlist)):
    print(i)
    stat = np.append(teamstats(int(idlist[i][0]),int(idlist[i][1])),teamstats(int(idlist[i][0]),int(idlist[i][2])))
    stats2.append(stat)
    
stats2 = np.array(stats2).astype(float)

#np.save('womenstatstest',stats2)

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

sub.to_csv('stage2women1.csv',header=['ID','Pred'],index=False)#change to stage 2

    
