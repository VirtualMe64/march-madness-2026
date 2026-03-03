# Manager to help load per game stats in a convenient format
# Created by Sammy Taubman

import warnings
import polars as pl
import os
from bisect import bisect_left

import numpy as np
from sklearn.preprocessing import StandardScaler

from .utils import split_games, add_basic_stats, add_four_factors
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

manager = None
def get_game_manager():
    global manager
    if manager is None:
        manager = GameManager("data", True)
    return manager

class GameManager:
    def __init__(self, data_path = "data", scale = True):
        self.m_games_reg = pl.read_csv(os.path.join(data_path, "MRegularSeasonDetailedResults.csv"))
        self.m_games_reg = split_games(self.m_games_reg).with_columns(pl.lit(1).alias('IsMens'))
        self.w_games_reg = pl.read_csv(os.path.join(data_path, "WRegularSeasonDetailedResults.csv"))
        self.w_games_reg = split_games(self.w_games_reg).with_columns(pl.lit(0).alias('IsMens'))
        self.games = pl.concat([self.m_games_reg, self.w_games_reg]).sort(['Season', 'TeamID', 'DayNum'])
        self.games = add_four_factors(add_basic_stats(self.games))

        self.all_cols = self.games.columns
        self.data_cols = [x for x in self.all_cols if x not in ['Season', 'DayNum', 'TeamID', 'OppTeamID']]
        scale_cols = [x for x in self.data_cols if x not in ['IsMens', 'NumOT', 'Result']]

        if scale:
            scaled_array = StandardScaler().fit_transform(self.games.select(scale_cols).to_numpy())
            for i, col in enumerate(scale_cols):
                self.games = self.games.with_columns(pl.Series(name=col, values=scaled_array[:, i]))

        # precompute data for faster access
        self.all_data = self.games.select(self.all_cols).to_numpy()

        # get start and end of each (Season/Team) section
        grouped = self.games.with_row_index(name='index').group_by(["Season", "TeamID"], maintain_order=True)
        self.min_indices = grouped.agg(pl.col('index').min()).partition_by(["Season", "TeamID"], include_key=False, as_dict=True)
        self.max_indices = grouped.agg(pl.col('index').max()).partition_by(["Season", "TeamID"], include_key=False, as_dict=True)
        for k in self.min_indices:
            self.min_indices[k] = self.min_indices[k].item()
            self.max_indices[k] = self.max_indices[k].item()
        self.indices = list(self.min_indices.keys())

    def get_games(self, season, team_id):
        target = (season, team_id)

        if target not in self.min_indices:
            return None
        return self.games[self.min_indices[target]:self.max_indices[target] + 1]

    def get_game(self, season, team_id, idx):
        target = (season, team_id)
        if target not in self.min_indices:
            return None
        return self.games[self.min_indices[target] + idx]

    def get_data(self, season, team_id, num_games = None, last_game_idx = None):
        '''
        Export games as an np array for model input
        Pads to ensure consistent size
        All games will be before last_game_idx
        '''
        target = (season, team_id)
        if target not in self.min_indices:
            return None, 0
        min_idx = self.min_indices[target]
        max_idx = self.max_indices[target] + 1
        # relevant games are in range [min_idx, max_idx]
        # we never want to fetch games >= last_game_idx
        # also we will never need more than num_games games
        max_idx = min(min_idx + last_game_idx, max_idx) if last_game_idx is not None else max_idx
        min_idx = max(min_idx, max_idx - num_games) if num_games is not None else min_idx
        data = self.all_data[min_idx:max_idx, :].copy()
        # if there's not enough games, pad
        # np.pad is slow so concat instead
        unpadded_len = len(data)
        if num_games is not None and len(data) < num_games:
            extra = num_games - len(data)
            data = np.vstack([np.zeros((extra, data.shape[1])), data])
        return data.astype(np.float32), unpadded_len

    def get_data_before_day(self, season, team_id, day_num, num_games = None):
        target = (season, team_id)
        if target not in self.min_indices:
            return None, 0
        min_idx = self.min_indices[target]
        max_idx = self.max_indices[target] + 1
        games = self.games[min_idx:max_idx]
        count = bisect_left(games['DayNum'].to_numpy(), day_num)

        max_idx = min(min_idx + count, max_idx)
        min_idx = max(min_idx, max_idx - num_games) if num_games is not None else min_idx
        data = self.all_data[min_idx:max_idx, :].copy()
        # if there's not enough games, pad
        # np.pad is slow so concat instead
        unpadded_len = len(data)
        if num_games is not None and len(data) < num_games:
            extra = num_games - len(data)
            data = np.vstack([np.zeros((extra, data.shape[1])), data])

        return data.astype(np.float32), unpadded_len

    def get_data_col_indices(self, cols):
        return [self.all_cols.index(x) for x in cols]

if __name__ == "__main__":
    import time
    
    start = time.time()
    gm = GameManager("./data", False)
    # ['NumOT', 'Result', 'Score', 'Home', 'Away', 'Neutral', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
    # 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppScore', 'OppFGM', 'OppFGA', 'OppFGM3', 'OppFGA3',
    # 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppPF', 'IsMens',
    # 'ScoreDifference', 'FGP', 'FGP3', 'FTP', 'FGA2', 'FGM2', 'FGP2', 'OppFGP', 'OppFGP3', 'OppFTP',
    # 'OppFGA2', 'OppFGM2', 'OppFGP2', 'EFGP', 'TOP', 'ORP', 'DRP', 'FTR', 'OppEFGP', 'OppTOP', 'OppORP',
    # 'OppDRP', 'OppFTR']

    print(f"Startup time: {time.time() - start}")
    start = time.time()
    for season in range(2003, 2026):
        for team in range(1200, 1300):
            x = gm.get_games(season, team) # georgia tech
    print(f"get_games: {time.time() - start}")
    start = time.time()

    for season in range(2003, 2020):
        for team in range(1300, 1364):
            x, _ = gm.get_data(season, team, 64) # georgia tech
    print(f"get_games_np: {time.time() - start}")
    start = time.time()

    for season in range(2003, 2020):
        for team in range(1300, 1364):
            x, _ = gm.get_data_before_day(season, team, 100, 64)
    print(f"get_games_np_before_day: {time.time() - start}")