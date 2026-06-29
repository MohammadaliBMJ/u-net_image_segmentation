import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from src.dataset import OxfordDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np

from src.model import UNet
from src.metrics import dice, intersection_of_union
from segmentation_models_pytorch import Unet


wandb.init(
    project="u-net_semantic_segmentation",
    config={
        "in_channels": 3,
        "num_classes": 3,
        "image_size": 128,
        "batch_size": 32,
        "lr": 1e-3,
    },
    dir='./outputs/'
)

pretrained = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if pretrained:
    model = Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
    )
else:
    model = UNet(in_channel=3, num_classes=3)

model = model.to(device)
wandb.watch(model)

# Data
train_data = OxfordDataset(root_path="./data/", split='trainval', image_size=128)
test_data = OxfordDataset(root_path="./data/", split='test', image_size=128)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Train
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)

if pretrained == True:
    epochs = 15
else:
    epochs = 25

lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)

print("Training starts...")


print(f"Training pretrained UNet: {pretrained}. training for {epochs} epochs.")
total_start_time = time.time()
for i in range(epochs):
    model.train()
    total_loss=0
    epoch_start_time = time.time()
    print(f"epoch {i} starts.")
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, masks)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

        # log lr
        current_lr = lr_scheduler.get_last_lr()[0]
        wandb.log({
            "lr": current_lr
        })

    
    # evaluation
    print(f"Evaluation in epoch {i} begins.")
    model.eval()
    val_loss = 0
    test_all_masks = []
    test_all_pred = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            output = model(images)
            loss = loss_function(output, masks)

            test_all_masks.append(masks)
            test_all_pred.append(output.argmax(dim=1))

            val_loss += loss.item()
    # visualize some test images.
    with torch.no_grad():
        idx = np.random.randint(0, len(test_data))
        image, mask = test_data[idx]

        unnorm_image = image.clone()
        unnorm_image = unnorm_image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + \
            torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        unnorm_image = (unnorm_image * 255).clamp(0, 255).byte()

        out_mask = model(image.unsqueeze(0).to(device)).argmax(dim=1)
        wandb.log({
            "mask_prediction": wandb.Image(
                unnorm_image.cpu().permute(1, 2, 0).numpy(),
                masks={
                    "prediction": {
                        "mask_data": out_mask[0].cpu().numpy()
                    },
                    "true_mask": {
                        "mask_data": mask.cpu().numpy()
                    }
                }
            )
        })

    test_all_masks = torch.cat(test_all_masks, dim=0)
    test_all_pred = torch.cat(test_all_pred, dim=0)

    iou = intersection_of_union(test_all_pred, test_all_masks, wandb.config.num_classes)
    dice_value = dice(test_all_pred, test_all_masks, wandb.config.num_classes)

    epoch_train_time = time.time() - epoch_start_time
    avg_epoch_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    wandb.log({
        "epoch": i,
        "train_loss": avg_epoch_loss,
        "val_loss": avg_val_loss,
        "epoch_training_time": epoch_train_time,
        "IoU": iou,
        "Dice": dice_value,
    })
    print(f"Epoch{i} finished. Training loss: {avg_epoch_loss}. Val loss: {avg_val_loss}. Epoch training time: {epoch_train_time}")

total_train_time = time.time() - total_start_time
print(f"Training is over. Total time: {total_train_time}")
wandb.run.summary["total_train_time"] = total_train_time
