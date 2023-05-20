from dataset import to_device, get_device, get_image
from model import get_model
import torch
import config


def predict_single():
    """
    This function predicts the class of a single image.

    Args:
        None

    Returns:
        The predicted class of the image.
    """

    device = get_device()
    image = get_image(config.PATH, device)
    model = get_model(device)
    model.eval()

    # Convert the image to a tensor and move it to the device.
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)

    # Make predictions.
    with torch.no_grad():
        preds = model(xb)

    # Get the top prediction.
    _, prediction = torch.max(preds.cpu().detach(), dim=1)

    # Decode the prediction.
    return decode_target(int(prediction), text_labels=True)


def decode_target(target, text_labels=True):
    """
    This function decodes a target class index into a string.

    Args:
        target: The target class index.
        text_labels: If True, the decoded class will be a string. If False, the decoded class will be an integer.

    Returns:
        The decoded class.
    """

    if text_labels:
        return config.IDX_CLASS_LABELS[target]
    else:
        return target


if __name__ == "__main__":
    print(predict_single())
