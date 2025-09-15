import os
import torch
import numpy as np
import imageio
from typing import Dict, Any
from zenml import step
from zenml.client import Client


@step
def log_and_checkpoint(
    epoch: int,
    models: Dict[str, torch.nn.Module],
    losses: Dict[str, float],
    inp: Dict[str, Any],
    out: Dict[str, Any],
    checkpoint_freq: int = 100,
    visualizer=None,
):
    """
    Logs metrics, saves checkpoints, and stores visualizations using ZenML.
    """

    run = Client().active_stack.experiment_tracker.get_run()  # e.g. MLflow or W&B

    # 1. Log metrics
    for name, value in losses.items():
        run.log_metric(f"{name}", float(value), step=epoch)

    # 2. Save checkpoint every N epochs
    if (epoch + 1) % checkpoint_freq == 0:
        cpk = {k: v.state_dict() for k, v in models.items()}
        cpk["epoch"] = epoch
        checkpoint_path = f"checkpoints/{str(epoch).zfill(6)}-checkpoint.pth.tar"
        torch.save(cpk, checkpoint_path)
        run.log_artifact(checkpoint_path, artifact_path="checkpoints")

    # 3. Save visualization (optional)
    if visualizer is not None:
        image = visualizer.visualize(inp["driving"], inp["source"], out)
        vis_path = f"visualizations/{str(epoch).zfill(6)}-rec.png"
        os.makedirs("visualizations", exist_ok=True)
        imageio.imsave(vis_path, image)
        run.log_artifact(vis_path, artifact_path="visualizations")

    return epoch