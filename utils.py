import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.losses = 0

    def update(self, val, n=1):
        self.count += n
        self.losses += val

    def show(self):
        return self.losses / self.count

def create_dir(_path):
    if not os.path.exists(_path):
        os.mkdir(_path)


def load_checkpoint(checkpoint, model, optimizer=None, lr_scheduler=None):
    print("=> Loading checkpoint")
    # print(checkpoint["state_dict"])
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        if checkpoint["lr_scheduler"] is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            print(f"")


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)




def plot_results(pred_masks, true_masks, images, result_path='./result_image', name=None):
    """
    Plot images, true masks, predicted masks and classes belonging to and save it to folder
    [Args]
    pred_masks: [tensor] shape BxCxHxW, predicted masks of model
    true_masks: [tensor] shape BxCxHxW, the true masks
    images: [PIL Image] shape CxHxW, True images
    result_path [str] the saved folder
    name [str] Name of file
    """

    for i in range(len(true_masks)):
        name = name if name is not None else 'temp'
        pred_masks_img = transforms.ToPILImage()(pred_masks[i]).convert('RGB')
        true_masks_img = transforms.ToPILImage()(true_masks[i]).convert('RGB')
        # images_img = transforms.ToPILImage()(images[i]).convert('RGB')

        fig = plt.figure(figsize=(16, 16))
        axes = []

        # True image
        axes.append(fig.add_subplot(1, 3, 1))
        axes[-1].title.set_text("Image")
        plt.axis('off')
        plt.imshow(images)

        # True mask
        axes.append(fig.add_subplot(1, 3, 2))
        axes[-1].title.set_text("True mask")
        plt.axis('off')
        plt.imshow(true_masks_img)

        # Predicted mask
        axes.append(fig.add_subplot(1, 3, 3))
        axes[-1].title.set_text("Pred mask ")
        plt.axis('off')
        plt.imshow(pred_masks_img)

        plt.savefig(os.path.join(result_path, name))
        plt.close(fig)   # Reusing the fig within each batch
    plt.close(fig)  # Clear the fig after each batch
