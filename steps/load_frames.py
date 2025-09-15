from zenml import step

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src.utils.frames_dataset import FramesDataset
from src.utils.frames_dataset import DatasetRepeater
# TO DO: Enable option to load 3 and 5 source frames
# TO DO: Experiment with random source frame selection and farthest point sampling 

@step
def load_dataset(config, is_train=True) -> DataLoader:
    train_params = config
    dataset = FramesDataset(is_train= is_train, **config['dataset_params'])
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    frames_data = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)
    return frames_data