import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix


def evaluate_predictions(all_labels, all_preds):
	acc = accuracy_score(all_labels, all_preds)
	f1 = f1_score(all_labels, all_preds, average="macro")

	print(f"Accuracy: {acc * 100:.2f}%")
	print(f"Macro F1: {f1:.4f}")
	print("\nClassification Report:")
	print(classification_report(all_labels, all_preds))

	cm = confusion_matrix(all_labels, all_preds, normalize="true")
	print("\nConfusion Matrix:")
	print(cm)

	return acc, f1, cm

def measure_inference_time(model, dataloader, device):
	model.eval()
	total_time = 0
	total_samples = 0

	with torch.no_grad():
		for images, _ in dataloader:
			images = images.to(device)

			if device.type == "cuda":
				torch.cuda.synchronize()
			
			start = time.time()
			_ = model(images)
			end = time.time()

			if device.type == "cuda":
				torch.cuda.synchronize()

		total_time += (end - start)
		total_samples += images.size(0)

	time_per_image = total_time / total_samples
	print(f"Time per image: {time_per_image:.6f} sec")

	return time_per_image
