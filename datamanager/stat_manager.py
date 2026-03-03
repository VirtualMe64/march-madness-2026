from .game_manager import get_game_manager
from .season_manager import get_season_manager
from abc import ABC, abstractmethod
import numpy as np

"""
Handle gathering statistics for a given team
Can handle both past game stats and aggregate stats
TODO: evaluate if these are neccesary or just a wrapper for the managers
"""

class StatManager(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_stats(self, season, team_id, day = None):
        pass

    @abstractmethod
    def metadata(self) -> list:
        pass

    @abstractmethod
    def min_games(self):
        pass

class GameStatManager(StatManager):
    def __init__(self, min_games = None, total_games = None, data_cols = None):
        self.manager = get_game_manager()
        self.min_games_val = min_games
        self.total_games = total_games
        self.data_indices = self.manager.get_data_col_indices(data_cols) if data_cols else \
                            self.manager.get_data_col_indices(self.manager.data_cols)

    def get_stats(self, season, team_id, day = None):
        if day is None:
            res, num = self.manager.get_data(season, team_id, num_games = self.total_games)
        else:
            res, num = self.manager.get_data_before_day(season, team_id, day, self.total_games)
        if res is None or num < self.min_games():
            return None
        return res[:, self.data_indices]

    def metadata(self):
        return [len(self.manager.data_cols) if self.data_indices is None else len(self.data_indices)]

    def min_games(self):
        return self.min_games_val if self.min_games_val else 0

class GameWithOpponentStatManager(StatManager):
    def __init__(self, min_games=None, total_games=None, game_cols=None, opp_cols=None, include_seed = False):
        self.manager = get_game_manager()
        self.aggregator = get_season_manager()
        self.min_games_val = min_games
        self.total_games = total_games
        self.include_seed = include_seed
        self.game_cols = game_cols
        self.game_indices = self.manager.get_data_col_indices(game_cols) if game_cols else \
                            self.manager.get_data_col_indices(self.manager.data_cols)
        self.opponent_cols = opp_cols
        self.opp_indices = self.aggregator.get_data_col_indices(opp_cols) if opp_cols else \
                            self.aggregator.get_data_col_indices(self.aggregator.data_cols)
        self.seed_col = self.aggregator.get_data_col_indices(["Seed"])[0]

    def get_stats(self, season, team_id, day=None):
        if day is None:
            res, num = self.manager.get_data(season, team_id, num_games = self.total_games)
        else:
            res, num = self.manager.get_data_before_day(season, team_id, day, self.total_games)
        if res is None or num < self.min_games():
            return None
        opponents = res[:, self.manager.all_cols.index("OppTeamID")]
        opponent_stats = np.array([
            self.aggregator.get_data(season, opp)[self.opp_indices] if opp != 0 else \
            np.zeros(len(self.opp_indices)).astype(np.float32)
            for opp in opponents
        ])
        res = res[:, self.game_indices]
        if self.include_seed:
            seed = self.aggregator.get_data(season, team_id)[self.seed_col]
            res = np.hstack([res, np.full((res.shape[0], 1), seed)])
        return np.hstack([res, opponent_stats])

    def metadata(self):
        game_size = len(self.manager.data_cols) if self.game_indices is None else len(self.game_indices)
        if self.include_seed:
            game_size += 1
        opp_size = len(self.aggregator.data_cols) if self.opp_indices is None else len(self.opp_indices)
        return [game_size, opp_size]

    def min_games(self):
        return self.min_games_val if self.min_games_val else 0

# todo: support non season wide stats
class AggStatManager(StatManager):
    def __init__(self, min_games = None, total_games = None, data_cols = None):
        if min_games is not None or total_games is not None:
            raise NotImplementedError("AggStatManager does not support min_games or total_games")
        self.manager = get_season_manager()
        self.data_indices = self.manager.get_data_col_indices(data_cols) if data_cols else None

    def get_stats(self, season, team_id, day = None):
        # AggStatManager currently only supports season wide stats
        res = self.manager.get_data(season, team_id)
        if res is None:
            return None
        return res[self.data_indices] if self.data_indices else res

    def metadata(self):
        return [len(self.manager.data_cols) if self.data_indices is None else len(self.data_indices)]

    def min_games(self):
        return 0


if __name__ == "__main__":
    stats = ["FTP", "FGP"]
    opp_stats = ["Seed"]
    gsm = GameStatManager(data_cols = stats)
    asm = AggStatManager(data_cols = stats)
    gwosm = GameWithOpponentStatManager(game_cols = stats, opp_cols = opp_stats)

    # print(get_game_manager().get_games(2024, 1210)[stats])
    # print(game_stat_manager.get_stats(2024, 1210))
    # print(get_game_manager().get_games(2024, 1210).mean()[stats])
    # print(agg_stat_manager.get_stats(2024, 1210))
    print(gwosm.get_stats(2024, 1210))