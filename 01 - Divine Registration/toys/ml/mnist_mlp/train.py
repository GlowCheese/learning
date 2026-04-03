from pathlib import Path

from fenn import Fenn
from fenn.utils import set_seed
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split

from .modules import get_trainer, test_dataset
from .modules import train_dataset as full_train_dataset

app = Fenn()
app.set_config_file(str(Path(__file__).parent / "fenn.yaml"))


# fmt: off

@app.entrypoint
def main(args):
    set_seed(args["train"]["seed"])  # set seed for reproducibility

    train_size = len(full_train_dataset)
    val_dataset, train_dataset = random_split(
        full_train_dataset, [
            train_size // 10,
            train_size - train_size // 10
        ]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["train"]["batch"],
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["test"]["batch"],
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args["val"]["batch"],
        shuffle=False
    )

    trainer = get_trainer(args)

    trainer.fit(train_loader=train_loader, val_loader=val_loader)

    y_test = [label for _, label in test_dataset]
    predictions = trainer.predict(test_loader)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")


# fmt: on

if __name__ == "__main__":
    app.run()
