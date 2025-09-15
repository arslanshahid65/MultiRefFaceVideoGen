### All workflows to be ported to ZenML based pipelines for better logging and tracking ###

from zenml import pipeline

from steps.load_frames import load_dataset
from steps.training import train

@pipeline(enable_cache=True)
def training_pipeline():
    frames_data = load_dataset()
    train(frames_data= frames_data)