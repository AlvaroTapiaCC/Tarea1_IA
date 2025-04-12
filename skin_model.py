import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class SkinDataset(Dataset):
    def __init__(self, filename):
        super(SkinDataset).__init__()
        self.data = np.load(filename)
        self.rgb = torch.tensor(self.data[:, :3], dtype=torch.float32)
        self.skin = torch.tensor(self.data[:, 3], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        return self.rgb[idx], self.skin[idx]

class SkinModel(nn.Module):
    def __init__(self, layer1, layer2, activation, output, classes):
        super(SkinModel, self).__init__()
        self.input_layer = nn.Linear(3, layer1)
        self.hidden_layer_1 = nn.Linear(layer1, layer2)
        self.output_layer = nn.Linear(layer2, classes)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if output == "sigmoid":
            self.output = nn.Sigmoid()
        elif output == "softmax":
            self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer_1(x))
        x = self.output_layer(x)
        return self.output(x)

class Trainer():
    def __init__(self, model, dataset, epochs, batch, device, loss_fn, optimizer, activation, output, sampler, threshold):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch = batch
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.activation = activation
        self.output = output
        self.sampler = sampler
        self.threshold = threshold
        self._prepare_data()
        self.model.to(self.device)

    def _prepare_data(self):
        n_train = int(len(self.dataset) * 0.8)
        n_val = len(self.dataset) - n_train
        train_set, val_set = random_split(self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        if self.sampler:
            labels = [int(label.item()) for _, label in train_set]
            class_sample_count = np.bincount(labels)
            class_weights = 1. / class_sample_count
            sample_weights = [class_weights[label] for label in labels]
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            self.train_loader = DataLoader(train_set, batch_size=self.batch, sampler=sampler)
        else:
            self.train_loader = DataLoader(train_set, batch_size=self.batch, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch, shuffle=False)

    def _calculate_accuracy(self, outputs, labels):
        if self.output == "sigmoid":
            preds = (outputs > self.threshold).float()
        else:
            preds = torch.argmax(outputs, dim=1).float()
        return (preds == labels).sum().item()

    def train(self):
        for epoch in range(self.epochs):
            acc = 0.0
            val_loss = 0.0
            epoch_loss = 0.0
            self.model.train()
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32 if self.output == "sigmoid" else torch.long)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() / len(self.train_loader)
                acc += self._calculate_accuracy(outputs, labels)
            with torch.no_grad():
                self.model.eval()
                for i, data in enumerate(self.val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.float32 if self.output == "sigmoid" else torch.long)
                    outputs = self.model(inputs)
                    val_loss += self.loss_fn(outputs, labels).item() / len(self.val_loader)
                    acc += self._calculate_accuracy(outputs, labels)

class Evaluator():
    def __init__(self, model, device, threshold):
        self.model = model
        self.device = device
        self.threshold = threshold

    def plot_roc_curve(self, true_labels, predicted_scores):
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='r', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def evaluate_images_with_masks(self, dataset_folder):
        self.model.eval()
        all_true = []
        all_scores = []
        print("Ruta completa usada:", dataset_folder)
        print("Archivos en la carpeta:")
        print(os.listdir(dataset_folder))
        image_filenames = sorted([f for f in os.listdir(dataset_folder) if f.lower().endswith('.jpg') and '_mask' not in f.lower()])
        print("Imágenes encontradas:")
        print(image_filenames)
        with torch.no_grad():
            for img_file in image_filenames:
                mask_file = img_file.replace('.jpg', '_mask.png')
                img_path = os.path.join(dataset_folder, img_file)
                mask_path = os.path.join(dataset_folder, mask_file)
                print(f"Procesando: {img_file} y {mask_file}")
                if not os.path.exists(mask_path):
                    print("No se encontró la máscara:", mask_path)
                    continue
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                print("Shape de la máscara:", mask.shape)
                print("Valores únicos en la máscara:", np.unique(mask))
                h, w, _ = image.shape
                pixels = image.reshape(-1, 3).astype(np.float32)
                pixels_tensor = torch.tensor(pixels, dtype=torch.float32).to(self.device)
                outputs = self.model(pixels_tensor)
                if outputs.dim() == 2 and outputs.shape[1] == 2:
                    outputs = outputs[:, 1]
                outputs = outputs.squeeze().cpu().numpy()
                labels = (mask.reshape(-1) > 127).astype(np.uint8)
                all_scores.extend(outputs)
                all_true.extend(labels)
        all_true_np = np.array(all_true)
        all_scores_np = np.array(all_scores)
        if len(np.unique(all_true_np)) < 2:
            print("Error: las etiquetas verdaderas no contienen ambas clases (0 y 1).")
            print("Valores encontrados en y_true:", np.unique(all_true_np))
            return
        self.plot_roc_curve(all_true_np, all_scores_np)
    


