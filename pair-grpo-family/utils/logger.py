import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(log_dir, "tb"))
        self.csv_path = os.path.join(log_dir, "reward.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "reward"])

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_reward(self, step, rew):
        self.csv_writer.writerow([step, rew])
        self.csv_file.flush()

    def close(self):
        self.writer.close()
        self.csv_file.close()
