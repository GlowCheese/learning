from torchvision import datasets, transforms

transform = transforms.ToTensor()

# fmt: off

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

# fmt: on
