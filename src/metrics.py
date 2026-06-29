import torch


def intersection_of_union(pred: torch.Tensor, mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute the mean Intersection over Union (mIoU) across all classes.

    Args:
        pred (torch.Tensor): Predicted segmentation map of shape (B, H, W) with class indices.
        mask (torch.Tensor): Ground-truth segmentation map of shape (B, H, W) with class indices.
        num_classes (int): Total number of classes to evaluate IoU over.

    Returns:
        float: Mean IoU computed by averaging per-class IoU values.
    """
    iou = 0.0
    for i in range(num_classes):
        pred_mask = (pred == i)
        ground_truth_mask = (mask == i)

        overlap = (pred_mask & ground_truth_mask).sum().float()
        union = pred_mask.sum().float() + ground_truth_mask.sum().float() - overlap

        if union == 0:
            iou += 1.0
        else:
            iou += overlap/union
    
    return (iou/num_classes).item()

def dice(pred: torch.Tensor, mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute the mean Dice coefficient across all classes.

    Args:
        pred (torch.Tensor): Predicted segmentation map of shape (B, H, W) with class indices.
        mask (torch.Tensor): Ground-truth segmentation map of shape (B, H, W) with class indices.
        num_classes (int): Total number of classes to evaluate Dice over.

    Returns:
        float: Mean Dice score computed by averaging per-class Dice values.
    """
    dice = 0.0
    for i in range(num_classes):
        pred_mask = (pred == i)
        ground_truth_mask = (mask == i)

        overlap = (pred_mask & ground_truth_mask).sum().float()
        denom = pred_mask.sum().float() + ground_truth_mask.sum().float()
        if denom == 0:
            dice += 1.0
        else:
            dice += (2 * overlap)/denom
    
    return (dice/num_classes).item()