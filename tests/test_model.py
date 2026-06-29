import torch
from src.model import UNet


def test_unet_forward_pass():
    """
    Ensure the UNet forward pass works and returns the correct output shape.

    Args:
        None

    Returns:
        None
    """
    model = UNet(in_channel=3, num_classes=4)
    x = torch.randn(1, 3, 128, 128)

    out = model(x)

    assert isinstance(out, torch.Tensor), "Output must be a torch.Tensor"
    assert out.shape == (1, 4, 128, 128), f"Unexpected output shape: {out.shape}"

def test_unet_has_parameters():
    """
    Ensure the UNet contains trainable parameters.

    Args:
        None

    Returns:
        None
    """
    model = UNet(in_channel=3, num_classes=4)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params > 0, "UNet should have trainable parameters but has zero"

def test_unet_backward_pass():
    """
    Ensure gradients flow through the UNet during backpropagation.

    Args:
        None

    Returns:
        None
    """
    model = UNet(in_channel=3, num_classes=4)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)

    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]

    assert len(grads) > 0, "No gradients found — backward pass failed"
