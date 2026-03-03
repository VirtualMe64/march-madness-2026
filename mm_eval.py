import polars as pl
import torch
from datamanager import GameManager
from utils import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser("Submission generation")
parser.add_argument("model", type=str, help="Path to the model")
parser.add_argument("config", type=str, help="Path to the config")
args = parser.parse_args()
cfg = parse_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CUTOFF = 2020
m_tourney_results = pl.read_csv('data/MNCAATourneyCompactResults.csv').filter(pl.col('Season') >= CUTOFF)
w_tourney_results = pl.read_csv('data/WNCAATourneyCompactResults.csv').filter(pl.col('Season') >= CUTOFF)
tourney_results = pl.concat([m_tourney_results, w_tourney_results]).sort(['Season', 'WTeamID', 'LTeamID'])

stats = cfg.stats

model = torch.jit.load(args.model).to(device)
model.eval()

acc = AverageMeter()
loss = AverageMeter()

for row in tqdm(tourney_results.rows(named=True)):
    team1 = min(row['WTeamID'], row['LTeamID'])
    team2 = max(row['WTeamID'], row['LTeamID'])
    result = 1 if row['WTeamID'] == team1 else 0
    data1 = stats.get_stats(row['Season'], team1, row['DayNum'])
    if data1 is None:
        continue
    data1 = torch.from_numpy(data1).unsqueeze(0)
    data2 = stats.get_stats(row['Season'], team2, row['DayNum'])
    if data2 is None:
        continue
    data2 = torch.from_numpy(data2).unsqueeze(0)
    
    pred = model(data1.to(device), data2.to(device))
    pred = pred.detach().numpy().item()
    if cfg.output == OutputType.LOGITS:
        pred = torch.sigmoid(pred)
    acc.update(pred > 0.5 if result else pred < 0.5)
    loss.update((result - pred) ** 2)

print(f"Accuracy: {acc.avg()}")
print(f"Loss: {loss.avg()}")