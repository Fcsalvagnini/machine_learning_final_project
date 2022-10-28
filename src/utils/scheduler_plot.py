import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

from schedulers import get_scheduler
from utils.configurator import SchedulerConfigs

with open("experiment_configs/nn_unet_nvidia.yaml") as yaml_file:
    experiment_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

cfg = SchedulerConfigs(experiment_configs["train_configs"]["scheduler"])
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = get_scheduler(
    cfg.scheduler_fn, 
    optimizer, 
    from_monai=cfg.from_monai, 
    **cfg.scheduler_kwargs)
lrs = []


for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
    scheduler.step()

fig = plt.figure()
plt.plot(lrs)
fig.savefig("scheduler.png")
