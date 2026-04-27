import torch
from PIL import Image
import torchvision.transforms as transforms
from src.models.resnet50 import get_model

classes = ["elliptical", "spiral", "lenticular", "irregular"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = get_model()
model.load_state_dict(torch.load("resnet50_galaxy.pth"))
model.eval()

image = Image.open("data/images_224/587722981742084144.jpg")

image = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
print("Predicted class:", classes[predicted.item()])

import matplotlib.pyplot as plt

plt.imshow(image.squeeze().permute(1,2,0))
plt.title(classes[predicted.item()])
plt.show()