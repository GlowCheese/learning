import torch
import torch.nn as nn
import torch.optim as optim
from fenn.nn import Trainer

from .model import MNISTMLP


def get_trainer(args: dict):
    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"

    # fmt: off

    model = MNISTMLP()
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
        epochs=args["train"]["epochs"],
        device=device,

        # for exporting model
        return_model="best",
        save_best=True,
        checkpoint_dir=args["checkpoint"]["dir"],
        checkpoint_name=args["checkpoint"]["name"],
    )

    # fmt: on

    return trainer
