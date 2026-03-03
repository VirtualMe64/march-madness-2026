import polars as pl

CONSTANT_STATS = ['Season', 'DayNum', 'NumOT']
TEAM_STATS = ['TeamID', 'Score', 'Home', 'Away', 'Neutral', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

DERIVED_CONSTANT_STATS = ['ScoreDifference']
DERIVED_TEAM_STATS = ['FGP', 'FGP3', 'FTP', 'FGA2', 'FGM2', 'FGP2']
FOUR_FACTORS = ['EFGP', 'TOP', 'ORP', 'DRP', 'FTR']

def add_four_factors(df : pl.DataFrame):
    # https://www.basketball-reference.com/about/factors.html
    for x in ['', 'Opp']:
        other = 'Opp' if x == '' else ''
        df = df.with_columns(
            ((pl.col(x + "FGM") + 0.5 * pl.col(x + "FGM3")) / pl.col(x + "FGA")).alias(x + "EFGP"),
            (pl.col(x + "TO") / (pl.col(x + "FGA") + 0.44 * pl.col(x + "FTA") + pl.col(x + "TO"))).alias(x + "TOP"),
            (pl.col(x + "OR") / (pl.col(x + "OR") + pl.col(other + "DR"))).alias(x + "ORP"),
            (pl.col(x + "DR") / (pl.col(x + "DR") + pl.col(other + "OR"))).alias(x + "DRP"),
            (pl.col(x + "FTM") / pl.col(x + "FGA")).alias(x + "FTR")
        )
    return df

def add_basic_stats(df : pl.DataFrame) -> pl.DataFrame:
    """
    Take a polars df with given stats and add percentages, two point stats, and score difference
    """
    df = df.with_columns(
        (pl.col('Score') - pl.col('OppScore')).alias("ScoreDifference")
    )

    for x in ['', 'Opp']:
        # calculate shooting percents
        # also derive 2 point shooting metrics
        df = df.with_columns(
            pl.when(pl.col(x + 'FGA') == 0).then(0).otherwise(pl.col(x + 'FGM') / pl.col(x + 'FGA')).alias(x + 'FGP'),
            pl.when(pl.col(x + 'FGA3') == 0).then(0).otherwise(pl.col(x + 'FGM3') / pl.col(x + 'FGA3')).alias(x + 'FGP3'),
            pl.when(pl.col(x + 'FTA') == 0).then(0).otherwise(pl.col(x + 'FTM') / pl.col(x + 'FTA')).alias(x + 'FTP'),
            (pl.col(x + 'FGA') - pl.col(x + 'FGA3')).alias(x + 'FGA2'),
            (pl.col(x + 'FGM') - pl.col(x + 'FGM3')).alias(x + 'FGM2')
        )
        df = df.with_columns(
            pl.when(pl.col(x + 'FGM2') == 0).then(0).otherwise(pl.col(x + 'FGM2') / pl.col(x + 'FGA2')).alias(x + 'FGP2'),
        )
    # filter out all infinite or nan values
    df = df.filter(~pl.any_horizontal(pl.selectors.numeric().is_infinite()))
    df = df.filter(~pl.any_horizontal(pl.selectors.numeric().is_nan()))

    return df

def split_games(game_df : pl.DataFrame) -> pl.DataFrame:
    """
    Input df has stats for several games containing winner stats and loser stats
    Split each game into two, each having team stats and opponent stats
    """
    game_df = game_df.with_columns(
        pl.when(pl.col('WLoc') == 'H').then(1).otherwise(0).alias('WHome'),
        pl.when(pl.col('WLoc') == 'A').then(1).otherwise(0).alias('WAway'),
        pl.when(pl.col('WLoc') == 'N').then(1).otherwise(0).alias('WNeutral'),
        pl.when(pl.col('WLoc') == 'N').then(1).otherwise(0).alias('LHome'),
        pl.when(pl.col('WLoc') == 'A').then(1).otherwise(0).alias('LAway'),
        pl.when(pl.col('WLoc') == 'H').then(1).otherwise(0).alias('LNeutral')
    )

    curr_cols = CONSTANT_STATS + ['W' + x for x in TEAM_STATS] + ['L' + x for x in TEAM_STATS]
    w_team_cols = CONSTANT_STATS + [x for x in TEAM_STATS] + ['Opp' + x for x in TEAM_STATS]
    l_team_cols = CONSTANT_STATS + ['Opp' + x for x in TEAM_STATS] + [x for x in TEAM_STATS]

    # map from winner stats to team1 and loser to team 2, augment, and add result column
    w_team_stats = game_df.select(curr_cols).rename({curr_cols[i] : w_team_cols[i] for i in range(0, len(curr_cols))})
    w_team_stats = w_team_stats.with_columns(pl.lit(1).alias("Result"))

    l_team_stats = game_df.select(curr_cols).rename({curr_cols[i] : l_team_cols[i] for i in range(0, len(curr_cols))})
    l_team_stats = l_team_stats.with_columns(pl.lit(0).alias("Result"))

    order = list(w_team_stats.columns)
    [order.remove(x) for x in ['OppAway', 'OppHome', 'OppNeutral']]

    return pl.concat([
        w_team_stats.select(order),
        l_team_stats.select(order)]
    )