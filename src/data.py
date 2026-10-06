from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "data/chest_xray"

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_loaders(batch_size=32):
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", train_tf)
    val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val", eval_tf)
    test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test", eval_tf)
    return (
        train_ds,
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size, num_workers=0),
        DataLoader(test_ds, batch_size, num_workers=0),
    )