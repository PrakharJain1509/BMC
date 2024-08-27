import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.mobilenet_v2(pretrained=False)
num_classes = 20
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('../camera_view_classification_model.pth'))
model = model.to(device)
model.eval()

# Define the data transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names (make sure this list matches the class names in your training dataset)
class_names = sorted(os.listdir('../dataset'))

# Function to predict the class of an image
def predict_image(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the device
    image = image.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    # Get the predicted class name
    predicted_class = class_names[preds.item()]
    return predicted_class

# Example usage
image_path = '../dataset/Kuvempu_Circle_FIX_2_time_2024-05-14T07-30-02_000/frame_3.jpg'
predicted_class = predict_image(image_path)
print(f'The predicted class is: {predicted_class}')
