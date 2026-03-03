import os
import polars as pl
from tqdm import tqdm
from torch.utils.data import Dataset
from .stat_manager import StatManager

MIN_SEASON = 2003

def load_games(data_path = "data", regular_season = False, post_season = False, seasons = None):
    assert regular_season or post_season
    dfs = []
    if regular_season:
        dfs.append(os.path.join(data_path, "MRegularSeasonCompactResults.csv"))
        dfs.append(os.path.join(data_path, "WRegularSeasonCompactResults.csv"))
    if post_season:
        dfs.append(os.path.join(data_path, "MNCAATourneyCompactResults.csv"))
        dfs.append(os.path.join(data_path, "WNCAATourneyCompactResults.csv"))
    combined = pl.concat([pl.read_csv(df) for df in dfs]).sort("Season")
    if seasons:
        combined = combined.filter(pl.col("Season").is_in(seasons))
    return combined.filter(pl.col("Season") >= MIN_SEASON)

class GameDataset(Dataset):
    def __init__(self, games: pl.DataFrame, stat_manager: StatManager):
        self.examples = []
        for game in tqdm(games.iter_rows(named=True), total=len(games)):
            season = game["Season"]
            team1 = game["WTeamID"]
            team2 = game["LTeamID"]
            team1_stats = stat_manager.get_stats(season, team1, game['DayNum'])
            team2_stats = stat_manager.get_stats(season, team2, game['DayNum'])
            if team1_stats is None or team2_stats is None:
                continue
            self.examples.append((team1_stats, team2_stats, 1))
            self.examples.append((team2_stats, team1_stats, 0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

if __name__ == "__main__":
    import time
    from .game_manager import GameManager
    from .stat_manager import GameStatManager, AggStatManager

    initial = time.time()
    games = load_games(regular_season=False, post_season=True)
    game_manager = GameManager("./data", scale=False)
    stat_manager = AggStatManager(manager = game_manager, season_wide=True)
    dataset = GameDataset(games, stat_manager)

    games, other, target = dataset[0]

    iteration_start = time.time()
    print(f"Startup time: {iteration_start - initial}")
    wins = 0
    losses = 0
    for i in range(len(dataset)):
        games, other, target = dataset[i]
        if target:
            wins += 1
        else:
            losses += 1
    
    print(f"Wins: {wins}, Losses: {losses}")

    end = time.time()