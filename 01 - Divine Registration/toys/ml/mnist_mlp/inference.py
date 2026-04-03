from pathlib import Path

import matplotlib.pyplot as plt
import torch
from fenn import Fenn

from .modules import get_trainer, test_dataset

app = Fenn()
app.set_config_file(str(Path(__file__).parent / "fenn.yaml"))


@app.entrypoint
def main(args):
    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"

    trainer = get_trainer(args)
    trainer.load_checkpoint(args["checkpoint"]["path"])

    selected_index = 500  # any number from 0 to 9999
    image, label = test_dataset[selected_index]

    tensor = image.to(device).unsqueeze(dim=0)
    pred = trainer._model(tensor).argmax(dim=1).item()
    verdict = "CORRECT :D" if label == pred else "INCORRECT T_T"

    print("Image label:", label)
    print("Predicted label:", pred)
    print("Prediction verdict:", verdict)

    plt.imshow(image.numpy().squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    app.run()
