import torch
import torch.nn as nn
from torch.utils.data import random_split

import sys
import math

from dataset import UniversalDataset, UniversalDataLoader
from model import Model, CheckpointHelper
from expmgr import expmgr

from typing import *


def import_task_train_and_test(
    task: str,
) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """
    Dynamically imports the train and test functions from the corresponding task module.

    Args:
        task (str): Task name (e.g., "zero_shot", "image_retrieval").

    Returns:
        Tuple containing the train and test functions.
    """
    sys.path.append("tasks")
    if task == "zero_shot":
        from zero_shot import train, test
    elif task == "image_retrieval":
        from image_retrieval import train, test
    return train, test


def main() -> None:
    """
    Main experiment runner.

    Loads settings from the ExperimentManager, prepares the dataset, dataloaders, model,
    optimizer, and runs training (optionally testing).
    """
    with expmgr.ExperimentManager() as settings:

        # 1. Load experiment configuration
        mode = settings["mode"]
        task = settings["task"]
        train, test = import_task_train_and_test(task)

        # 2. Set seed and device
        torch.manual_seed(settings["misc"]["seed"])
        device = torch.device(settings["misc"]["device"])

        # 3. Load training hyperparameters
        batch_size: int = settings[mode]["batch_size"]
        learning_rate = float(settings[mode]["learning_rate"])

        # 4. Dataloader parameters
        shuffle: bool = settings["dataloader"]["shuffle"]
        num_workers: int = settings["dataloader"]["num_workers"]

        # 5. Dataset loading
        dataset_name: str = settings["dataset"]["name"]
        dataset_info = settings["dataset_infos"][dataset_name]

        dataset = UniversalDataset(dataset_name, **dataset_info)

        # Set forget rule for filtering out forget set samples (0 -> 19)
        dataset.set_forget_rule(eval(settings["dataset"]["forget_rule"]))

        # 6. Split dataset into training and testing (70% train / 30% test)
        train_num_samples = math.floor(0.7 * len(dataset))
        train_test_num_samples = [train_num_samples, len(dataset) - train_num_samples]
        train_set, test_set = random_split(dataset, train_test_num_samples)

        # 7. Create data loaders
        train_loader = UniversalDataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        test_loader = UniversalDataLoader(
            test_set, batch_size=settings["val"]["batch_size"], shuffle=False, num_workers=num_workers
        )

        # 8. Initialize model and optimizer
        model = Model().to(device)
        weight_decay = float(settings[mode]["weight_decay"])
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=weight_decay,
        )

        # 9. Set up checkpoint helper to save best models
        checkpoint_helper = CheckpointHelper()

        # 10. Begin training
        print(f"[*] Bat dau huan luyen voi task: {task.upper()} | Dataset: {dataset_name} | Epochs: {settings[mode]['epochs']}")
        train(model, train_loader, optimizer, settings, checkpoint_helper, val_loader=test_loader)


if __name__ == "__main__":
    main()