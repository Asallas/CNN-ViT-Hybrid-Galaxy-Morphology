import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

from src.datasets.dataset import GalaxyDataset
from metrics_utils import evaluate_predictions

def evaluate_hybrid(cnn, vit, dataloader, device, threshold):
    cnn.eval()
    vit.eval()

    all_labels = []
    all_preds = []

    total_time = 0
    total_samples = 0
    vit_calls = 0

    use_cuda = device.type == "cuda"


    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if use_cuda:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                starter.record()

                cnn_outputs = cnn(images)
                probs = torch.softmax(cnn_outputs, dim=1)
                conf, preds = probs.max(dim=1)

                final_preds = preds.clone()
                mask = conf < threshold

                if mask.any():
                    vit_inputs = images[mask]
                    vit_outputs = vit(vit_inputs)
                    vit_preds = vit_outputs.argmax(dim=1)
                    final_preds[mask] = vit_preds
                    vit_calls += mask.sum().item()

                ender.record()
                torch.cuda.synchronize()

                elapsed = starter.elapsed_time(ender) / 1000.0
            else:
                start = time.time()

                cnn_outputs = cnn(images)
                probs = torch.softmax(cnn_outputs, dim=1)
                conf, preds = probs.max(dim=1)

                final_preds = preds.clone()
                mask = conf < threshold

                if mask.any():
                    vit_inputs = images[mask]
                    vit_outputs = vit(vit_inputs)
                    vit_preds = vit_outputs.argmax(dim=1)
                    final_preds[mask] = vit_preds
                    vit_calls += mask.sum().item()

                end = time.time()
                elapsed = end - start
            
            total_time += (elapsed)
            total_samples += images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())

    acc, f1, cm = evaluate_predictions(all_labels, all_preds)
    time_per_img = total_time / total_samples
    vit_usage = vit_calls / total_samples

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Time per image: {time_per_img:.6f} sec")
    print(f"VIT %: {vit_usage * 100:.2f}%")

    return acc, f1, cm, time_per_img, vit_usage

def main(cnn, vit, cnn_path, vit_path, csv_path, image_dir, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.load_state_dict(torch.load(cnn_path))
    vit.load_state_dict(torch.load(vit_path))

    cnn.to(device)
    vit.to(device)

    dataset = GalaxyDataset(csv_path, image_dir, transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=8)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for thresh in thresholds:
        start = time.time()
        acc, f1, cm, time_per_img, vit_usage = evaluate_hybrid(cnn, vit, loader, device, thresh)
        end = time.time()
        timer = start - end
        print(f"Final execution time:{timer:.6f}sec")
        results.append( {"threshold": thresh, "accuracy": acc, "f1": f1, "cm": cm, "time_per_img": time_per_img, "vit_usage": vit_usage} )

    print("\nSummary of Results:")
    for r in results:
        print(r)

    return results

if __name__ == "__main__":
    from src.models.resnet50 import get_model
    from src.models.vit import get_vit_model
    import torchvision.transforms as transforms
    cnn = get_model()
    vit = get_vit_model()
    
    cnn_path = "Models/resnet50_galaxy3_best.pth"
    vit_path = "Models/vit_galaxy3_best.pth"
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

    results = main(cnn, vit, cnn_path, vit_path, csv_path, img_dir, transform)
    
    
