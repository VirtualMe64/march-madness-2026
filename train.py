import os
import time
import torch
from torch.utils.data import DataLoader
from datamanager import *
from utils import *

# todo: return val/train loss/accuracy as list for plotting
def train(cfg : Config, verbose = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats = cfg.stats
    train_games = load_games(regular_season=cfg.reg_season, post_season=cfg.post_season, seasons=list(range(2000, 2020)))
    val_games = load_games(regular_season=cfg.reg_season, post_season=cfg.post_season, seasons=list(range(2020, 2026)))
    train = GameDataset(train_games, stats)
    val = GameDataset(val_games, stats)
    datapoints = []
    dataloaders = {'train': DataLoader(train, batch_size=cfg.train_batch, shuffle=True),
                   'val': DataLoader(val, batch_size=cfg.val_batch, shuffle=False)}
    model = cfg.model(*cfg.stats.metadata(), **cfg.args).to(device)

    print(f"Training on {len(train)} examples")
    print(f"Validating on {len(val)} examples")
    print(f"{sum(m.numel() for m in model.parameters())} parameters")

    epochs = cfg.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = cfg.loss
    output_to_pred = {
        OutputType.LOGITS: lambda x : x > 0,
        OutputType.PROBS: lambda x : x > 0.5,
    }[cfg.output]

    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()
    val_accuracy_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    for epoch in range(epochs):
        model.train()
        accuracy_meter.reset()
        loss_meter.reset()
        val_accuracy_meter.reset()
        val_loss_meter.reset()
        if verbose:
            print("Training")
        start = time.time()
        for i, (x1, x2, y) in enumerate(dataloaders['train']):
            optimizer.zero_grad()
            output = model(x1.to(device), x2.to(device))
            loss = criterion(output, y.unsqueeze(1).float().to(device))
            loss.backward()

            optimizer.step()
            
            preds = output_to_pred(output)
            acc = (preds == y.unsqueeze(1).to(device)).float().mean()
            accuracy_meter.update(acc.item(), len(y))
            loss_meter.update(loss.item(), len(y))
            if verbose == 0:
                print(f"\tEpoch {epoch}, Batch {i}, Loss: {loss.item()}, Accuracy: {acc.item()}, Time: {time.time() - start}")
            start = time.time()
        if verbose:
            print("Validation")
        model.eval()
        with torch.no_grad():
            for i, (x1, x2, y) in enumerate(dataloaders['val']):
                output = model(x1.to(device), x2.to(device))
                loss = criterion(output, y.unsqueeze(1).float().to(device))
                preds = output_to_pred(output)
                acc = (preds == y.unsqueeze(1).to(device)).float().mean()
                val_accuracy_meter.update(acc.item(), len(y))
                val_loss_meter.update(loss.item(), len(y))
                if verbose == 0:
                    print(f"\tEpoch {epoch}, Batch {i}, Loss: {loss.item()}, Accuracy: {acc.item()}")

        datapoints.append((loss_meter.avg(), accuracy_meter.avg(), val_loss_meter.avg(), val_accuracy_meter.avg()))
        print(f"Epoch {epoch}, Loss: {loss_meter.avg()}, Accuracy: {accuracy_meter.avg()}, Val Loss: {val_loss_meter.avg()}, Val Accuracy: {val_accuracy_meter.avg()}")

    print("Saving to ", f"checkpoints/{cfg.name}.pt")
    model.eval()
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    torch.jit.save(torch.jit.script(model), f"checkpoints/{cfg.name}.pt")

    return datapoints

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("Train script")
    parser.add_argument("config", help="path to .yaml for training configuration", type=str)
    args = parser.parse_args()

    cfg = parse_config(args.config)
    data = train(cfg, False)[int(cfg.epochs * 0.1):]

    loss = [x[0] for x in data]
    accuracy = [x[1] for x in data]
    val_loss = [x[2] for x in data]
    val_accuracy = [x[3] for x in data]

    plt.figure()
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()

    plt.figure()
    plt.plot(accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.legend()

    plt.show()