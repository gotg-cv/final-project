import torch

from src.device_utils import device_name, get_torch_device
from src.model_builder import get_daisee_model


def main():
    device = get_torch_device()
    print(f"Device: {device_name(device)}")

    model = get_daisee_model(freeze_base=True)
    model.to(device)
    model.eval()

    x = torch.randn(1, 16, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(pixel_values=x)

    assert out.logits.shape == (1, 4), out.logits.shape
    print(f"OK — logits {tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
