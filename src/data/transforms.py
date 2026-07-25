from torchvision import transforms

def get_train_transforms(img_size=128, advanced=False):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
    ]
    if advanced:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.2,  # ~±20% như paper
                contrast=0.2,
            ),
        ]
    else:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,
                scale=(0.9, 1.1),  # zoom 0.9–1.1 như paper
            ),
            transforms.ColorJitter(brightness=0.2),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    return transforms.Compose(base)

def get_eval_transforms(img_size=128):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])