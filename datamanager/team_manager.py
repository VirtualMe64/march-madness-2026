# Manager to help with names and ids of teams
# Created by Sammy Taubman

import pandas as pd
import os
import re

def canonize(team_name):
    s = team_name.strip().lower()
    s = re.sub('[^0-9a-z\s]+', '', s).lower()
    return s

class TeamManager:
    def __init__(self, data_path):
        m_team_data = pd.read_csv(os.path.join(data_path, "MTeams.csv"))
        w_team_data = pd.read_csv(os.path.join(data_path, "WTeams.csv"))

        self.team_by_id = pd.concat([m_team_data, w_team_data], axis = 0).set_index("TeamID")

        self.m_team_spellings = pd.read_csv(os.path.join(data_path, "MTeamSpellings.csv"))
        self.m_team_spellings['TeamNameSpelling'] = self.m_team_spellings['TeamNameSpelling'].map(canonize)
        self.m_team_spellings = self.m_team_spellings.set_index('TeamNameSpelling')
        self.w_team_spellings = pd.read_csv(os.path.join(data_path, "WTeamSpellings.csv"))
        self.w_team_spellings['TeamNameSpelling'] = self.w_team_spellings['TeamNameSpelling'].map(canonize)
        self.w_team_spellings = self.w_team_spellings.set_index('TeamNameSpelling')

    def id_to_name(self, team_id):
        '''
        Convert a teams id to it's formatted name. I.e 1181 -> 'Duke'.
        Returns `None` if id is invalid
        '''
        if team_id not in self.team_by_id.index:
            return None
        res = self.team_by_id.loc[team_id]['TeamName']
        assert type(res) == str # in case there's somehow multiple results
        return res

    def name_to_id(self, team_name, is_mens):
        '''
        Convert a teams name to it's id. I.e 'duke', is_mens = True -> 1181.
        Applies preprocessing to ignore case and non-alpha characters
        Gender must be specified (since i.e duke mens and duke womens have different ids)
        Returns `None` if name can't be found
        '''
        source = self.m_team_spellings if is_mens else self.w_team_spellings
        target = canonize(team_name)

        if target not in source.index:
            return None
        res = int(source.loc[target]['TeamID'])
        return res

if __name__ == "__main__":
    tm = TeamManager("../data")
    print(tm.id_to_name(1181)) # duke
    print(tm.name_to_id('duke', True))  # 1181
    print(tm.name_to_id('duke', False)) # 3181