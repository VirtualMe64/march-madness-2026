import yaml
import os
import torch
from torch import nn
from dataclasses import dataclass
from enum import Enum
import importlib
from datamanager import StatManager, AggStatManager, GameStatManager, GameWithOpponentStatManager

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.count = 0
    
    def update(self, val, count = 1):
        self.val += val * count
        self.count += count
    
    def avg(self):
        return self.val / self.count if self.count > 0 else 0

class OutputType(Enum):
    LOGITS = 0
    PROBS = 1    

@dataclass
class Config:
    epochs: int
    learning_rate: int
    weight_decay: int
    train_batch: int
    val_batch: int
    name: str
    model: nn.Module
    args: dict
    reg_season: bool
    post_season: bool
    stats: StatManager
    output: OutputType
    loss: nn.Module

def parse_config(path : str) -> Config:
    '''
    Parse all neccesary information for model training or evaluation
    '''
    torch
    assert os.path.exists(path), f"Invalid path, {path}"
    with open(path, "r") as open_file:
        try:
            data = yaml.safe_load(open_file)
        except Exception as exc:
            raise exc
    cfg = Config(
        epochs = _load(data, "Train/epochs", int, 10),
        learning_rate = _load(data, "Train/learning_rate", float, 0.01),
        weight_decay = _load(data, "Train/weight_decay", float, 0),
        train_batch = _load(data, "Train/train_batch", int, 128),
        val_batch = _load(data, "Train/val_batch", int, 128),
        name = _load(data, "Model/name", str),
        model = _load_model(_load(data, "Model/module", str), _load(data, "Model/name", str)),
        args = _load(data, "Model/args", dict, {}),
        reg_season = _load(data, "Data/reg_season", bool, False),
        post_season = _load(data, "Data/post_season", bool, False),
        stats = _load_stats(data),
        output = _load_enum(data, "Loss/output", {
            'Logits': OutputType.LOGITS,
            'Probs': OutputType.PROBS
        }),
        loss = _load_enum(data, "Loss/loss", {
            'BCE': nn.BCELoss,
            'BCEWithLogits': nn.BCEWithLogitsLoss,
            'MSE': nn.MSELoss
        })()
    )
    return cfg

def _load(data, path, type, default = ...):
    """
    Helper function to load a value from a yaml dictionary
    If any key in the path doesn't exist, throws an error
    Example path: Train/epochs
    """
    curr = data
    for part in path.split("/"):
        if part not in curr:
            if default is not ...:
                return default
            raise Exception(f"Config is missing argument: {path}")
        curr = curr[part]
    return type(curr)

def _load_enum(data, path, enum_dict, default = ...):
    key = _load(data, path, str, default)
    if key not in enum_dict:
        raise Exception(f"Config argument {path} of {key} is invalid. Must be one of {list(enum_dict.keys())}")
    return enum_dict[key]

def _load_model(module : str, name : str) -> nn.Module:
    """
    Load a torch model given path to module and name of model class
    """
    module = getattr(importlib.import_module(module), name)
    return module

def _load_stats(data : dict):
    agg = _load(data, "Data/Agg/active", bool, False)
    game = _load(data, "Data/Game/active", bool, False)
    game_opp = _load(data, "Data/GameWithOpponent/active", bool, False)
    assert sum([agg, game, game_opp]) == 1, "Only one stat loader can be active at a time"
    if agg:
        args = _load(data, "Data/Agg", dict, {})
        args.pop("active")
        return AggStatManager(**args)
    elif game:
        args = _load(data, "Data/Game", dict, {})
        args.pop("active")
        return GameStatManager(**args)
    elif game_opp:
        args = _load(data, "Data/GameWithOpponent", dict, {})
        args.pop("active")
        return GameWithOpponentStatManager(**args)
    else:
        raise ValueError("No stat loader detected")