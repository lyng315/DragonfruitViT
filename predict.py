import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse

CHECKPOINTS_DIR = "checkpoints"

from models.vgg16_model import get_vgg16_model
from models.resnet50_model import get_resnet50_model
from models.mobilenetv2_model import get_mobilenetv2_model
from models.vit_model import get_vit_model

def predict(image_path, model_name, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['DomNau', 'KhoeManh', 'NamCanh', 'ThanThu', 'ThoiBe']
    num_classes = len(class_names)

    if model_name == 'vgg16':
        model = get_vgg16_model(num_classes)
    elif model_name == 'resnet50':
        model = get_resnet50_model(num_classes)
    elif model_name == 'mobilenetv2':
        model = get_mobilenetv2_model(num_classes)
    elif model_name == 'vit':
        model = get_vit_model(num_classes)
    else:
        print("Invalid model name.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    print(f"\n{'='*20} INFERENCE RESULT {'='*20}")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Prediction: {class_names[predicted_idx]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    print(f"{'='*57}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Dragon Fruit Disease Classification")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    
    m_path = args.path if args.path else os.path.join(CHECKPOINTS_DIR, f"best_{args.model}.pth")
    
    if os.path.exists(args.image) and os.path.exists(m_path):
        predict(args.image, args.model, m_path)
    else:
        print("Error: Image or model checkpoint not found.")
