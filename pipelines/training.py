### All workflows to be ported to ZenML based pipelines for better logging and tracking ###

from zenml import pipeline

from steps.load_frames import load_dataset
from steps.train import train

@pipeline
def train_pipeline():
    frames_data = load_dataset()
    train(frames_data= frames_data)