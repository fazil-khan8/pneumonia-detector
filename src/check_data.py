from data import get_loaders

train_ds, train_loader, val_loader, test_loader = get_loaders()

print("Classes:", train_ds.classes)
print("Train images:", len(train_loader.dataset))
print("Val images:", len(val_loader.dataset))
print("Test images:", len(test_loader.dataset))

x, y = next(iter(train_loader))
print("Batch shape:", x.shape)