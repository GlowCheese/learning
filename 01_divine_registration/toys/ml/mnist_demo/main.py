import matplotlib.pyplot as plt
import torch
from rich import print

# from tqdm import tqdm
from rich.progress import track
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("---------------")

epochs = 5
batch_size = 64
_inspect_data = False


#######################
# LOADING DATASETS
#######################

# fmt: off

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# fmt: on


#######################
# INSPECTION
#######################


if _inspect_data:
    print("# training samples:", len(train_dataset))
    print("# test samples:", len(test_dataset))
    print("---------------")

    print("42nd image's label:", train_dataset[41][1])
    print("showing the 42nd image:")

    temp_img = train_dataset[41][0].numpy().squeeze()
    plt.imshow(temp_img, cmap="gray")
    plt.show()
    print("---------------")

    temp_images, temp_labels = next(iter(train_loader))
    print("input shape (images):", temp_images.shape)
    print("output shape (labels):", temp_labels.shape)
    print("---------------")


#######################
# MODEL DEFINITION
#######################


class MNISTMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(784, 128)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        # fmt: off
        y = x.view(x.shape[0], -1)  # (batch_size, 784)
        y = self.linear_1(y)        # (batch_size, 128)
        y = self.relu(y)            # (batch_size, 128)
        y = self.linear_2(y)        # (batch_size, 10)
        # fmt: on
        return y


#######################
# TRAIN THE MODEL
#######################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTMLP().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# report data for visualization
accs = []
losses = []


for epoch in track(range(epochs), description="Training..."):
    # train epoch
    tot_loss = 0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = tot_loss / len(train_loader)
    losses.append(avg_loss)

    # eval on test
    correct = 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / len(test_dataset)
    accs.append(acc)

    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}")
