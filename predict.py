from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
args = parser.parse_args()

base_dir = Path(__file__).resolve().parent
model_file = base_dir / "model_output" / "coating_defect_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model = torch.load(model_file, map_location=device, weights_only=False)
labels = saved_model["labels"]
thresholds = saved_model["thresholds"]
image_size = saved_model["image_size"]

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(saved_model["model_state"])
model = model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = Image.open(args.image).convert("RGB")
image_tensor = image_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    probabilities = torch.sigmoid(model(image_tensor))[0].cpu().numpy()

print(f"Image: {args.image}")
for label, probability, threshold in zip(labels, probabilities, thresholds):
    prediction = "yes" if probability >= threshold else "no"
    print(f"{label:16s} probability={probability:.3f} defect={prediction}")
