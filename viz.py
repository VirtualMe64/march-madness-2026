import torch
from utils import parse_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = parse_config("configs/attention.yaml")
stats = cfg.stats
model = cfg.model(*cfg.stats.metadata(), **cfg.args).to(device)
model.load_state_dict(torch.jit.load("checkpoints/Attention.pt").state_dict())
model.eval()


x1 = torch.from_numpy(stats.get_stats(2024, 1210)).unsqueeze(0).to(device)
x2 = torch.from_numpy(stats.get_stats(2024, 1210)).unsqueeze(0).to(device)
print(model(x1, x2))