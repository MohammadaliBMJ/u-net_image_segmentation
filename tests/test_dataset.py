import torch
from src.dataset import OxfordDataset

def test_dataset_first_item():
    """
    Verify that OxfordDataset:
    - initializes correctly
    - contains at least one sample
    - returns an image tensor of shape (3, 128, 128) with dtype float32
    - returns a mask tensor of shape (128, 128) with dtype int64
    """
    ds = OxfordDataset(root_path="../data/", split="trainval", image_size=128)

    assert len(ds) > 0, "Dataset length should be > 0 but got an empty dataset"

    image, mask = ds[0]

    # image checks
    assert isinstance(image, torch.Tensor), "Image must be a torch.Tensor"
    assert image.shape == (3, 128, 128), f"Image shape must be (3,128,128) but got {image.shape}"
    assert image.dtype == torch.float32, f"Image dtype must be float32 but got {image.dtype}"

    # mask checks
    assert isinstance(mask, torch.Tensor), "Mask must be a torch.Tensor"
    assert mask.shape == (128, 128), f"Mask shape must be (128,128) but got {mask.shape}"
    assert mask.dtype == torch.int64, f"Mask dtype must be int64 but got {mask.dtype}"

def test_dataset_resizing():
    """
    Verify that changing image_size produces correctly resized
    image and mask tensors.
    """
    ds = OxfordDataset(root_path="../data/", split="trainval", image_size=64)

    image, mask = ds[0]

    assert image.shape == (3, 64, 64), f"Expected image shape (3,64,64), got {image.shape}"
    assert mask.shape == (64, 64), f"Expected mask shape (64,64), got {mask.shape}"

def test_mask_class_ids(dataset):
    """
    Ensure mask contains only valid class IDs (0–3).
    This prevents training crashes due to unexpected labels.
    """
    _, mask = dataset[0]

    unique_vals = torch.unique(mask)
    for v in unique_vals:
        assert 0 <= v <= 3, f"Mask contains invalid class ID: {v.item()}"
