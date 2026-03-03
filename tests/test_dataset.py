from datamanager import GameManager, GameDataset, GameStatManager, load_games
import numpy as np
import polars as pl

manager = GameManager("data", scale=False)
games = load_games(regular_season=True)
games = games.filter(
    (pl.col("Season") == 2024) &
    (
        (pl.col("WTeamID") == 1210) | (pl.col("LTeamID") == 1210)
    )
)

# 2024 georgia tech, https://www.espn.com/mens-college-basketball/team/schedule/_/id/59/season/2024
# 32 game season
dataset30 = GameDataset(games, GameStatManager(manager, 30, 30))
dataset31 = GameDataset(games, GameStatManager(manager, 31, 31))
dataset32 = GameDataset(games, GameStatManager(manager, 32, 32))

def test_dataset_length():
    # game against utah in both directions
    assert len(dataset30) == 4
    assert sorted([dataset30[0][2], dataset30[1][2], dataset30[2][2], dataset30[3][2]]) == [0, 0, 1, 1]
    assert len(dataset31) == 2
    assert sorted([dataset31[0][2], dataset31[1][2]]) == [0, 1]
    assert len(dataset32) == 0

def test_data():
    game = manager.get_game(2024, 1210, 31) # 32th game (against utah)
    game_day = game.item(0, "DayNum")
    result = game.item(0, "Result")
    winner = game.item(0, "TeamID") if result == 1 else game.item(0, "OppTeamID")
    loser = game.item(0, "TeamID") if result == 0 else game.item(0, "OppTeamID")
    expected_data, _ = manager.get_data_before_day(2024, winner, game_day, 31)
    expected_opp, _ = manager.get_data_before_day(2024, loser, game_day, 31)

    assert np.allclose(dataset31[0][0], expected_data)
    assert np.allclose(dataset31[0][1], expected_opp)
    assert np.allclose(dataset31[1][0], expected_opp)
    assert np.allclose(dataset31[1][1], expected_data)
    assert 1 == dataset31[0][2]
    assert 0 == dataset31[1][2]
    
    expected_data, _ = manager.get_data_before_day(2024, winner, game_day, 30)
    expected_opp, _ = manager.get_data_before_day(2024, loser, game_day, 30)
    assert np.allclose(dataset30[2][0], expected_data)
    assert np.allclose(dataset30[3][0], expected_opp)
    assert np.allclose(dataset30[2][1], expected_opp)
    assert np.allclose(dataset30[3][1], expected_data)
    assert 1 == dataset30[2][2]
    assert 0 == dataset30[3][2]

    game = manager.get_game(2024, 1210, 30) # 31th game (against virginia)
    game_day = game.item(0, "DayNum")
    result = game.item(0, "Result")
    winner = game.item(0, "TeamID") if result == 1 else game.item(0, "OppTeamID")
    loser = game.item(0, "TeamID") if result == 0 else game.item(0, "OppTeamID")
    expected_data, _ = manager.get_data_before_day(2024, winner, game_day, 30)
    expected_opp, _ = manager.get_data_before_day(2024, loser, game_day, 30)
    assert np.allclose(dataset30[0][0], expected_data)
    assert np.allclose(dataset30[1][0], expected_opp)
    assert np.allclose(dataset30[0][1], expected_opp)
    assert np.allclose(dataset30[1][1], expected_data)
    assert 1 == dataset30[0][2]
    assert 0 == dataset30[1][2]