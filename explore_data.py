from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data" / "raw" / "CoatingVision" / "classification"
labels = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]

data = pd.read_csv(data_dir / "labels.csv")

print(f"Number of images: {len(data)}")
print("\nNumber of images per label:")
print(data[labels].sum())

label_counts = data[labels].sum()
label_counts.plot.bar(figsize=(8, 4))
plt.ylabel("Number of images")
plt.title("Label distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(base_dir / "reports" / "figures" / "label_distribution.png", dpi=160)
plt.close()

plt.figure(figsize=(12, 3))
for index, label in enumerate(labels):
    example = data[data[label] == 1].iloc[0]
    image = Image.open(data_dir / "images" / example["file_name"])

    plt.subplot(1, 4, index + 1)
    plt.imshow(image)
    active_labels = [name for name in labels if example[name] == 1]
    plt.title(" + ".join(active_labels), fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig(base_dir / "reports" / "figures" / "sample_images.png", dpi=160)
plt.close()

