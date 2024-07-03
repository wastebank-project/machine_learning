import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load the custom-trained YOLOv8s model
model = YOLO('YOLOv8s_e100_lr0.0001_32.pt')

# Define the classes (assuming your model was trained with the same classes)
class_names = [
    'Botol Kaca',
    'Botol Plastik',
    'Galon',
    'Gelas Plastik',
    'Kaleng',
    'Kantong Plastik',
    'Kantong Semen',
    'Kardus',
    'Kemasan Plastik',
    'Kertas Bekas',
    'Koran',
    'Pecahan Kaca',
    'Toples Kaca',
    'Tutup Galon'
]

# Function to load an image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

# Load your image
image_path = 'botol dan galon.png'  # Update with the path to your image
original_image = load_image(image_path)

# Perform detection
results = model(original_image)

# Visualize the image and the bounding boxes
def plot_predictions(image, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for result in results[0].boxes:  # Iterate through detected objects
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        score = result.conf[0].cpu().numpy()
        label = class_names[int(result.cls[0].cpu().numpy())]
        if score > 0.5:  # Only plot if the score is above a threshold
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f'{label} : {score:.2f}', bbox=dict(facecolor='yellow', alpha=1))

    plt.axis('off')
    plt.show()

plot_predictions(original_image, results)