from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import torch


class OxfordDataset(Dataset):
    """Dataset wrapper for Oxford-IIIT Pet segmentation.

    Loads images and masks, applies image and mask transforms, and returns
    tensors ready for training.
    """
    def __init__(self, root_path: str, split='trainval', image_size=128):
        """Initialize dataset and define transforms.

        Args:
            root_path (str): Path where the dataset is stored or downloaded.
            split (str, optional): Dataset split to use. Defaults to 'trainval'.
            image_size (int, optional): Resize target for image and mask. Defaults to 128.
        """
        super().__init__()

        self.dataset = OxfordIIITPet(
            root=root_path,
            split=split,
            target_types='segmentation',
            download=True,
        )

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self):
        """Return number of samples."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Load and transform one (image, mask) pair.

        Args:
            idx (int): Sample index.

        Returns:
            torch.Tensor: Image tensor (C,H,W) float32.
            torch.Tensor: Mask tensor (H,W) int64 with class IDs.
        """
        image, mask = self.dataset[idx]
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = mask - 1

        return image, mask