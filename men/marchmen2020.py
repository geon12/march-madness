"""

"""

import numpy as np
np.random.seed(12)

import pandas as pd

from os import getcwd

direc = getcwd()


seasonres = pd.read_csv(direc+'/Stage2UpdatedDataFiles/RegularSeasonDetailedResults.csv')

seeds = pd.read_csv(direc+'/Stage2UpdatedDataFiles/NCAATourneySeeds.csv')

res = pd.read_csv(direc+'/Stage2UnchangedDataFiles/NCAATourneyCompactResults.csv')


def noofevents():

    rand = np.random.normal(loc=230,scale=20)
    return int(rand)
    


def statcalc(team1,year):
    team1stat = seasonres[seasonres['WTeamID']==team1][seasonres['Season']==year]
    team1stat['WOR'] = team1stat['WOR']/(team1stat['WOR']+team1stat['LDR'])
    team1stat['WDR'] = team1stat['WDR']/(team1stat['WDR']+team1stat['LOR'])
    team1stat['WStl'] = team1stat['WStl']/team1stat['LTO']
    team1stat['WBlk'] = team1stat['WBlk'] /team1stat['LFGA']
    team1stat['WPF'] = team1stat['WPF'] /(team1stat['LTO']+team1stat['LFGA']+team1stat['LFTA'])
    #team1stat['WTO'] = team1stat['WTO']/(team1stat['WFGA']+team1stat['WTO']+team1stat['WFTA'])
    team1stat['OTO'] = team1stat['LTO']/(team1stat['LTO']+team1stat['LFGA']+team1stat['LFTA'])
    team1stat = team1stat[['WFGM','WFGA','WFGM3','WFGA3','WFTM','WFTA','WOR','WDR','WAst','WTO','WStl','WBlk','WPF','OTO']] 
    team1statb = seasonres[seasonres['LTeamID']==team1][seasonres['Season']==year]
    team1statb['LOR'] = team1statb['LOR']/(team1statb['LOR']+team1statb['WDR'])
    team1statb['LDR'] = team1statb['LDR']/(team1statb['LDR']+team1statb['WOR'])
    team1statb['LStl'] = team1statb['LStl']/team1statb['WTO']
    team1statb['LBlk'] = team1statb['LBlk'] /team1statb['WFGA']
    team1statb['LPF'] = team1statb['LPF']/(team1statb['WTO']+team1statb['WFGA']+team1statb['WFTA'])
    #team1statb['LTO'] = team1statb['LTO']/(team1statb['LFGA']+team1statb['LTO']+team1statb['LFTA'])
    team1statb['OTO'] = team1statb['WTO']/(team1statb['WTO']+team1statb['WFGA']+team1statb['WFTA'])
    team1statb = team1statb[['LFGM','LFGA','LFGM3','LFGA3','LFTM','LFTA','LOR','LDR','LAst','LTO','LStl','LBlk','LPF','OTO']]
    
    team1stat.columns = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF','OTO']
    team1statb.columns = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF','OTO']
    team1stat = pd.concat([team1stat,team1statb])
    
    team1stat['pos'] = team1stat['FGA']+team1stat['TO']+team1stat['FTA']
    team1stat['TO'] = team1stat['TO']/team1stat['pos']
    team1stat['FGA2'] = team1stat['FGA']/team1stat['pos']
    team1stat['FTA2'] = team1stat['FTA']/team1stat['pos']
    team1to = np.mean(team1stat['TO'])
    team1fta = np.mean(team1stat['FTA2'])
    team1fga = np.mean(team1stat['FGA2'])
    team1poss= np.asarray([team1fga,team1fta,team1to])
    team1poss[team1poss<0] = .001
    team1poss = team1poss/np.sum(team1poss)
    
    team1takeaway = np.mean(team1stat['OTO'])
    team1pf = np.mean(team1stat['PF'])
    team1nothing = 1 - team1takeaway-team1pf
    team1def= np.asarray([team1takeaway,team1pf,team1nothing])
    team1def[team1def<0] = .001
    team1def = team1def/np.sum(team1def)
    
    freethrowperc1 = team1stat['FTM']/(team1stat['FTM']+team1stat['FTA']) 
    freethrowperc1 = np.mean(freethrowperc1)
    team1block = np.mean(team1stat['Blk'])
    
    team1_2or3 = team1stat['FGA3']/team1stat['FGA']
    team1_2or3 = np.mean(team1_2or3)
    
    perctwoteam1 = (team1stat['FGM']-team1stat['FGM3'])/((team1stat['FGA']-team1stat['FGA3']))
    perctwoteam1 = np.mean(perctwoteam1)
    
    perc3team1 = team1stat['FGM3']/team1stat['FGA3']
    perc3team1 = np.mean(perc3team1)
    
    team1or = np.mean(team1stat['OR'])
    team1dr = np.mean(team1stat['DR'])
    
    return [team1poss,team1def, freethrowperc1,team1block,team1_2or3,perctwoteam1,perc3team1,team1or,team1dr]
    
def eventAct(turn,score1,team2def,team2block,team2dr,team1poss,team1def, freethrowperc1,team1block,team1_2or3,perctwoteam1,perc3team1,team1or,team1dr):
    defense =np.random.choice(['takeaway','pf','nothing'],p=team2def)
    if(defense=='takeaway'):
        turn=1
    if(defense!='takeaway'):
        if defense=='pf':
            rand = np.random.random()
            if(rand<=freethrowperc1):
                score1 = score1 + 1
                turn = 1
            else:
                randor = np.random.random()*team1or
                randdr = np.random.random()*team2dr
                if(randor>randdr):
                    turn =0
                else:
                    turn =1
        else:
            
            offense = np.random.choice(['shoot','freethrow','turnover'],p=team1poss)
            if(offense=='turnover'):
                turn=1
            if offense=='freethrow':
                
                rand = np.random.random()
                if(rand<=freethrowperc1):
                    score1 = score1 + 1
                    turn = 1
                else:
                    randor = np.random.random()*team1or
                    randdr = np.random.random()*team2dr
                    if(randor>randdr):
                        turn =0
                    else:
                        turn =1
            if(offense=='shoot'):
                rand = np.random.random()
                if(rand<=team2block):
                    recover = np.random.choice([0,1],p=[.5,.5])
                    turn = recover
                else:
                    twoorthree = np.random.choice([2,3],p=[1-team1_2or3,team1_2or3])
                    if twoorthree ==2:
                        rand = np.random.random()
                        if(rand<=perctwoteam1):
                            score1 = score1 +2
                            turn =1
                        else:
                            randor = np.random.random()*team1or
                            randdr = np.random.random()*team2dr
                            if(randor>randdr):
                                turn =0
                            else:
                                turn =1
                    else:
                        rand = np.random.random()
                        if(rand<=perc3team1):
                            score1 = score1 +3
                            turn =1
                        else:
                            randor = np.random.random()*team1or
                            randdr = np.random.random()*team2dr
                            if(randor>randdr):
                                turn =0
                            else:
                                turn =1
    return turn, score1

def simulation(team1,team2,adj1,adj2,year,a,b):
    
    """
    Team 1 is always the lower seed
    """
    
    score1 = 0
    score2 = 0
    
    eventno = noofevents()
    

    team1poss,team1def, freethrowperc1,team1block,team1_2or3,perctwoteam1,perc3team1,team1or,team1dr = a
    team2poss,team2def, freethrowperc2,team2block,team2_2or3,perctwoteam2,perc3team2,team2or,team2dr = b
    
    turn = np.random.randint(0,2)#whose possession it is
    
    for i in range(0,eventno):
        if turn==0:
            turn,score1 = eventAct(turn,score1,team2def,team2block,team2dr,team1poss,team1def, freethrowperc1,team1block,team1_2or3,perctwoteam1,perc3team1,team1or,team1dr)
        else:
            turn,score2 = eventAct(turn,score2,team1def,team1block,team1dr,team2poss,team2def, freethrowperc2,team2block,team2_2or3,perctwoteam2,perc3team2,team2or,team2dr)
            turn = 1-turn

    score1 = score1 +adj1
    score2 = score2 +adj2
    
    if score1==score2:
        tiebreaker = np.random.randint(0,2)
        if tiebreaker==1:
            score2 +=1
        else:
            score1 += 1
    if score2>score1:

        return 0

    if score1>score2:

        return 1

massey = pd.read_csv("MasseyOrdinals_2018_133_only_43Systems.csv")#change ranking daynum?

def multiSim(team1,team2,year):
    numofsim=100
    
    massey2 = massey[['TeamID','OrdinalRank']][massey['SystemName']=='POM'][massey['Season']==year][massey['RankingDayNum']==133]
    rank1 = massey2['OrdinalRank'][massey2['TeamID']==team1].values[0]
    rank2 =  massey2['OrdinalRank'][massey2['TeamID']==team2].values[0]
    adjscore1 = -60/350*rank1 +(10560/350)
    adjscore2 = -60/350*rank2 +(10560/350)
    count = 0
    a = statcalc(team1,year)
    b = statcalc(team2,year)
    for i in range(0,numofsim):
            
        count +=simulation(team1,team2,adjscore1,adjscore2,year,a,b)
    
   
    count = count/ numofsim#currently prob of team1
    
    return count


sub = pd.read_csv('SampleSubmissionStage2.csv')#change to stage 2

idlist = sub['ID'].tolist()
idlist = [x.split('_') for x in idlist]



prob = []
for i in range(0,len(idlist)):
    print(i)
    
    count = multiSim(int(idlist[i][1]),int(idlist[i][2]),int(idlist[i][0]))
    if(count==0):
        count = .01
    if(count==1):
        count = .99
    prob.append(count)
    
sub['Pred'] = prob

sub.to_csv('stage2men1.csv',header=['id','pred'],index=False)

    
