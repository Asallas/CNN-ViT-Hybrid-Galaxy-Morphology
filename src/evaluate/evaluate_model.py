import torch
from torch.utils.data import DataLoader
from dataset import GalaxyDataset
import pandas as pd
import time

from metrics_utils import evaluate_predictions

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    total_time = 0.0
    total_samples = 0

    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if use_cuda:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()
                outputs = model(images)
                ender.record()

                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender) / 1000.0  # ms → sec
            else:
                start = time.time()
                outputs = model(images)
                end = time.time()
                elapsed = end - start

            preds = outputs.argmax(dim=1)

            total_time += elapsed
            total_samples += images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    time_per_img = total_time / total_samples
    return all_labels, all_preds, time_per_img, total_time

def main(model, model_path, csv_path, image_dir, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    dataset = GalaxyDataset(csv_path, image_dir, transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=8)

    
    labels, preds, execution_time, total = evaluate(model, loader, device)
    

    acc, f1, cm = evaluate_predictions(labels, preds)

    return acc, f1, cm, execution_time, total

if __name__ == "__main__":
    from ResNet50Model import get_model
    from ViTModel import get_vit_model
    import torchvision.transforms as transforms
    model = get_model()
    model2 = get_vit_model()
    
    model_path = "Models/resnet50_galaxy3_best.pth"
    model_path2 = "Models/vit_galaxy3_best.pth"
    csv_path = "csv_files/test_labels2.csv"
    img_dir = "data/images3_224"

    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    acc, f1, cm, execution_time, total = main(model, model_path, csv_path, img_dir, transform)
    print(f"Time per image: {execution_time:.6f} sec")
    print(f"Final Accuracy: {acc * 100:.2f}%")
    print(f"Final Macro F1: {f1:.4f}")
