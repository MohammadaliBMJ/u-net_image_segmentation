import torch
from src.metrics import intersection_of_union
from src.metrics import dice


# Intersection of Union
def test_iou_perfect_match():
    """
    IoU should be 1.0 when prediction and ground truth are identical.
    """
    pred = torch.tensor([[0, 1],
                         [1, 0]])
    mask = torch.tensor([[0, 1],
                         [1, 0]])

    result = intersection_of_union(pred, mask, num_classes=2)
    assert result == 1.0, f"Expected IoU=1.0 for perfect match, got {result}"


def test_iou_no_overlap():
    """
    IoU should be 0.0 when prediction and ground truth have no overlapping pixels.
    """
    pred = torch.tensor([[0, 0],
                         [0, 0]])
    mask = torch.tensor([[1, 1],
                         [1, 1]])

    result = intersection_of_union(pred, mask, num_classes=2)
    assert result == 0.0, f"Expected IoU=0.0 for no overlap, got {result}"


def test_iou_empty_class():
    """
    IoU for a class absent in both pred and GT should be 1.0.
    """
    pred = torch.tensor([[0, 0],
                         [0, 0]])
    mask = torch.tensor([[0, 0],
                         [0, 0]])

    # class 0 present, class 1 absent → IoU = (IoU_0 + IoU_1) / 2 = (1 + 1) / 2 = 1
    result = intersection_of_union(pred, mask, num_classes=2)
    assert result == 1.0, f"Expected IoU=1.0 when class is absent in both pred and GT, got {result}"

# Dice
def test_dice_perfect_match():
    """
    Dice should be 1.0 when prediction and ground truth are identical.
    """
    pred = torch.tensor([[0, 1],
                         [1, 0]])
    mask = torch.tensor([[0, 1],
                         [1, 0]])

    result = dice(pred, mask, num_classes=2)
    assert result == 1.0, f"Expected Dice=1.0 for perfect match, got {result}"


def test_dice_no_overlap():
    """
    Dice should be 0.0 when prediction and ground truth have no overlapping pixels.
    """
    pred = torch.tensor([[0, 0],
                         [0, 0]])
    mask = torch.tensor([[1, 1],
                         [1, 1]])

    result = dice(pred, mask, num_classes=2)
    assert result == 0.0, f"Expected Dice=0.0 for no overlap, got {result}"


def test_dice_empty_class():
    """
    Dice for a class absent in both pred and GT should be 1.0.
    """
    pred = torch.tensor([[0, 0],
                         [0, 0]])
    mask = torch.tensor([[0, 0],
                         [0, 0]])

    # class 0 present, class 1 absent → Dice = (Dice_0 + Dice_1) / 2 = (1 + 1) / 2 = 1
    result = dice(pred, mask, num_classes=2)
    assert result == 1.0, f"Expected Dice=1.0 when class is absent in both pred and GT, got {result}"
