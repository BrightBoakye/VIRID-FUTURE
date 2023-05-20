import openai
from dataset import to_device, get_device, get_image
from model import get_model
import torch
import config

# Set up OpenAI API credentials
openai.api_key = "[SECRET-API-KEY]"

def get_insights_and_recommendations(land_type):
    """
    Get insights and recommendations for a given land type.

    Args:
        land_type (str): The land type to get insights and recommendations for.

    Returns:
        str: The insights and recommendations for the given land type.
    """

    prompt = f"What are the best practices for optimizing yield in {land_type} land type ?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.8,
    )
    recommendation = response.choices[0].text.strip()
    return recommendation


def predict_single(image):
    """
    Predict the land type of an image.

    Args:
        image (torch.Tensor): The image to predict the land type of.

    Returns:
        tuple(str, str): The predicted land type and insights and recommendations for the predicted land type.
    """

    device = get_device()
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    with torch.no_grad():
        preds = model(xb)
    _, prediction = torch.max(preds.cpu().detach(), dim=1)

    # Get insights and recommendations using OpenAI API
    land_type = decode_target(int(prediction), text_labels=True)
    insights_and_recommendations = get_insights_and_recommendations(land_type)

    return land_type, insights_and_recommendations


if __name__ == "__main__":
    image = get_image(config.PATH)
    land_type, insights_and_recommendations = predict_single(image)
    print(f"Predicted land type: {land_type}")
    print(f"Insights and recommendations: {insights_and_recommendations}")
else:
    print("Sorry no insights for this landtype")
