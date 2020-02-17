# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:45:40 2020

@author: dragos.munteanu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam

epl_1819 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1920/E0.csv")
epl_1819 = epl_1819[['HomeTeam','AwayTeam','FTHG','FTAG']]
epl_1819 = epl_1819.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

epl_1819_master = epl_1819
epl_test = epl_1819[-10:]
epl_actual = epl_test
epl_test = epl_test[['HomeTeam', 'AwayTeam']]
epl_1819 = epl_1819[:-10]

print(epl_1819.mean())

# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

goal_model_data = pd.concat([epl_1819[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl_1819[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()

def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

def predict_match(poisson_model, epl_test):
    pred = []
    #Iterating through each pair of maches in a dataframe
    for team1, team2 in epl_test.itertuples(index=False):
        
        #Running match simulation and calculating probabilities
        match = simulate_match(poisson_model, team1, team2, max_goals=10)
        
        pred.append({'Match': (team1,team2), 'HomeWin': np.sum(np.tril(match, -1)),
                'Draw': np.sum(np.diag(match)),
                'AwayWin': np.sum(np.triu(match, 1))})
    
    return pd.DataFrame(pred)





    