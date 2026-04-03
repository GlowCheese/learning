import torch
import torch.nn as nn
import torch.optim as optim
from fenn.nn import Checkpoint, Trainer

from .model import MNISTCNN


def get_trainer(args: dict):
    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"

    # fmt: off

    model = MNISTCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args["train"]["lr"])
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        num_classes=10,
        device=device,

        checkpoint_config=Checkpoint(
            dir=args["checkpoint"]["dir"],
            name=args["checkpoint"]["name"],
            epochs=4,
            save_best=True,
        ),
        early_stopping_patience=5
    )

    # fmt: on

    return trainer
