from dataset import to_device, get_device, get_image
from model import get_model
import torch
import config



def predict_single():
    device = get_device()
    image = get_image(config.PATH,device)
    model = get_model(device)
    model.eval()
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    with torch.no_grad():
        preds = model(xb)
    _, prediction = torch.max(preds.cpu().detach(), dim=1)
    return decode_target(int(prediction), text_labels=True)



def decode_target(target, text_labels=True):
    result = []
    if text_labels:
        return config.IDX_CLASS_LABELS[target]
    else:
        return target

if __name__ == "__main__":
    print(predict_single())