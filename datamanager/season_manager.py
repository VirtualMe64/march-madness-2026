# Manager to help load season wide state (seed, sos, mean stats) in a convenient format
# Created by Sammy Taubman

import os
import polars as pl
import numpy as np
from .utils import split_games, add_basic_stats, add_four_factors
from sklearn.preprocessing import StandardScaler
from functools import lru_cache

manager = None
def get_season_manager():
    global manager
    if manager is None:
        manager = SeasonManager("data", True)
    return manager

def get_seed_data(data_path):
    m_seed_data = pl.read_csv(f"{data_path}/MNCAATourneySeeds.csv")
    w_seed_data = pl.read_csv(f"{data_path}/WNCAATourneySeeds.csv")
    seed_data = pl.concat([m_seed_data, w_seed_data]).sort("Season")
    seed_data = seed_data.with_columns(
        pl.col("Seed").str.extract(r"(\d+)").cast(pl.Int64())
    )
    return seed_data

# todo: https://dpmartin42.github.io/posts/r/college-basketball-rankings#:~:text=The%20RPI%20is%20one%20of,'%20winning%20percentage%20(OOWP).
# todo: this is a pretty mid implementation rn
def add_sos(games : pl.DataFrame) -> pl.DataFrame:
    win_rate = games.group_by(["Season", "TeamID"]) \
                    .agg(pl.col("Result").mean().alias("SOS")) \
                    .rename({"TeamID": "OppTeamID"})
    games = games.join(win_rate, on=["Season", "OppTeamID"], how="left")
    games = games.with_columns(pl.col("SOS"))
    return games

# todo: this is pretty weird rn -- maybe just merge with GameManager
class SeasonManager:
    def __init__(self, data_path = "data", scale = True):
        m_games_reg = pl.read_csv(os.path.join(data_path, "MRegularSeasonDetailedResults.csv"))
        m_games_reg = split_games(m_games_reg).with_columns(pl.lit(1).alias('IsMens'))
        w_games_reg = pl.read_csv(os.path.join(data_path, "WRegularSeasonDetailedResults.csv"))
        w_games_reg = split_games(w_games_reg).with_columns(pl.lit(0).alias('IsMens'))
        games = pl.concat([m_games_reg, w_games_reg]).sort(['Season', 'TeamID', 'DayNum'])
        games = add_sos(games)
        games.drop_in_place("DayNum")
        games.drop_in_place("OppTeamID")
        seed_data = get_seed_data(data_path)
        games = games.join(seed_data, on=["Season", "TeamID"], how="left")
        games = games.with_columns(pl.col("Seed").fill_null(32)) # this may be bad

        counts = games.group_by(["Season", "TeamID"]).len()
        sum_stats = games.group_by(["Season", "TeamID"]).sum()
        sum_stats = sum_stats.join(counts, on=["Season", "TeamID"], how="left")
        sum_stats = add_four_factors(add_basic_stats(sum_stats))

        pct_stats = ['FGP', 'FGP2', 'FGP3', 'FTP', 'EFGP', 'TOP', 'OPR', 'DRP', 'FTR']
        pct_stats = ['Opp' + x for x in pct_stats] + pct_stats
        no_divide = ['Season', 'TeamID', 'Count'] + pct_stats
        sum_stats = sum_stats.with_columns([
            (pl.col(x) / pl.col('len')).alias(x) for x in sum_stats.columns if x not in no_divide
        ])
        sum_stats.drop_in_place('len')

        self.data_cols = [x for x in sum_stats.columns if x not in ['Season', 'TeamID']]
        scale_cols = [x for x in self.data_cols if x not in ['Result', 'NumOT', 'IsMens']]
        if scale:
            scaled_array = StandardScaler().fit_transform(sum_stats.select(scale_cols).to_numpy())
            for i, col in enumerate(scale_cols):
                sum_stats = sum_stats.with_columns(pl.Series(name=col, values=scaled_array[:, i]))
        self.stats = sum_stats.partition_by(['Season', 'TeamID'], as_dict=True)

    @lru_cache(350)
    def get_stats(self, season, team_id):
        if (season, team_id) not in self.stats:
            return None
        # todo: can this to_numpy be optimized?
        return self.stats[(season, team_id)]

    @lru_cache(350)
    def get_data(self, season, team_id):
        if (season, team_id) not in self.stats:
            return None
        # todo: can this to_numpy be optimized?
        return self.stats[(season, team_id)].select(self.data_cols).to_numpy()[0].astype(np.float32)

    def get_data_col_indices(self, cols):
        return [self.data_cols.index(x) for x in cols]


if __name__ == "__main__":
    m = SeasonManager(scale=True)

    print(m.get_stats(2024, 1163)['Seed'])
    print(m.get_data(2024, 1163)[m.get_data_col_indices(['Seed'])])