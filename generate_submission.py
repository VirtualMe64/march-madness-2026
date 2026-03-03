import torch
import os
from datamanager import GameManager
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from utils import *

parser = argparse.ArgumentParser("Submission generation")
parser.add_argument("model", type=str, help="Path to the model")
parser.add_argument("config", type=str, help="Path to the config")
args = parser.parse_args()
cfg = parse_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stats = cfg.stats

model = torch.jit.load(args.model).to(device)
model.eval()

submission = pd.read_csv(os.path.join("data", "SampleSubmissionStage1.csv"))
out = submission.copy()
submission['season'] = submission['ID'].str[0:4].astype(int)
submission['team1'] = submission['ID'].str[5:9].astype(int)
submission['team2'] = submission['ID'].str[10:14].astype(int)
submission.loc[len(submission)] = [0, 0, 0, 0, 0]

prev = None
others = []
start_idx = -1
shape = None
for row in tqdm(submission.itertuples(), total = submission.shape[0]):
    team1 = row[4]
    team2 = row[5]
    if team1 != prev:
        if prev is not None:
            # prediction logic
            data1 = stats.get_stats(row[3], prev, None)
            if data1 is None:
                data1 = np.zeros(shape).astype(np.float32)
            data1 = torch.from_numpy(data1)
            data2 = torch.cat(others)
            data1 = torch.cat([data1.unsqueeze(0) for _ in range(len(others))])
            # todo: above may be wrong (gotta think about agg vs seq case)
            pred = model(data1.to(device), data2.to(device))
            if cfg.output == OutputType.LOGITS:
                pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy().reshape(len(others))
            out.loc[start_idx:start_idx + len(others) - 1, 'Pred'] = pred
        others = []
        prev = team1
        start_idx = int(row[0])

    data2 = stats.get_stats(row[3], team2, None)
    if shape is None and data2 is not None:
        shape = data2.shape
    if data2 is None:
        data2 = np.zeros(shape).astype(np.float32)
    data2 = torch.from_numpy(data2).unsqueeze(0)
    others.append(data2)

out.to_csv("out.csv", index=False)