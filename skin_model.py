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
        super().__init__()  # Fixed super() call
        self.data = np.load(filename)
        # Normalize RGB values to [0,1] range
        self.rgb = torch.tensor(self.data[:, :3], dtype=torch.float32) / 255.0
        self.skin = torch.tensor(self.data[:, 3], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        return self.rgb[idx], self.skin[idx]

class SkinModel(nn.Module):
    def __init__(self, layer1, layer2, activation, output, classes):
        super(SkinModel, self).__init__()
        # Enhanced model with more layers and dropout
        self.model = nn.Sequential(
            nn.Linear(3, layer1),
            nn.BatchNorm1d(layer1),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Dropout(0.3),
            
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Dropout(0.3),
            
            nn.Linear(layer2, layer2 // 2),
            nn.BatchNorm1d(layer2 // 2),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Dropout(0.2),
            
            nn.Linear(layer2 // 2, classes)
        )
        
        if output == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = lambda x: x

    def forward(self, x):
        x = self.model(x)
        return self.output_activation(x)

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
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_samples = 0
            
            # Training loop
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32 if self.output == "sigmoid" else torch.long)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                correct = self._calculate_accuracy(outputs, labels)
                train_acc += correct
                train_samples += inputs.size(0)
            
            # Validation loop
            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.float32 if self.output == "sigmoid" else torch.long)
                    
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    correct = self._calculate_accuracy(outputs, labels)
                    val_acc += correct
                    val_samples += inputs.size(0)
            
            # Calculate epoch statistics
            train_loss /= train_samples
            train_acc /= train_samples
            val_loss /= val_samples
            val_acc /= val_samples
            
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Validation loss improved to {val_loss:.4f}")
                torch.save(self.model.state_dict(), "best_model.pth")
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load("best_model.pth"))

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
        print("Imágenes encontradas:", len(image_filenames))
        
        with torch.no_grad():
            for img_file in image_filenames:
                mask_file = img_file.replace('.jpg', '_mask.png')
                img_path = os.path.join(dataset_folder, img_file)
                mask_path = os.path.join(dataset_folder, mask_file)
                print(f"Procesando: {img_file}")
                
                if not os.path.exists(mask_path):
                    print("No se encontró la máscara:", mask_path)
                    continue
                
                # Read and preprocess image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    print(f"Error al cargar imagen o máscara: {img_path}")
                    continue
                    
                h, w, _ = image.shape
                
                # Normalize pixels to match training data normalization
                pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
                
                # Process in batches to handle batch normalization correctly
                batch_size = 1024
                total_pixels = pixels.shape[0]
                all_outputs = []
                
                for i in range(0, total_pixels, batch_size):
                    end = min(i + batch_size, total_pixels)
                    batch_pixels = pixels[i:end]
                    batch_tensor = torch.tensor(batch_pixels, dtype=torch.float32).to(self.device)
                    batch_outputs = self.model(batch_tensor)
                    
                    # Handle both binary and multi-class outputs
                    if batch_outputs.dim() == 2 and batch_outputs.shape[1] == 2:
                        batch_outputs = batch_outputs[:, 1]
                    batch_outputs = batch_outputs.squeeze().cpu().numpy()
                    all_outputs.append(batch_outputs)
                
                # Combine all batch outputs
                outputs = np.concatenate(all_outputs)
                
                # Create binary ground truth labels from mask
                labels = (mask.reshape(-1) > 127).astype(np.uint8)
                
                if len(outputs) != len(labels):
                    print(f"Error: Output and label sizes don't match: {len(outputs)} vs {len(labels)}")
                    continue
                    
                all_scores.extend(outputs)
                all_true.extend(labels)
        all_true_np = np.array(all_true)
        all_scores_np = np.array(all_scores)
        if len(np.unique(all_true_np)) < 2:
            print("Error: las etiquetas verdaderas no contienen ambas clases (0 y 1).")
            print("Valores encontrados en y_true:", np.unique(all_true_np))
            return
        self.plot_roc_curve(all_true_np, all_scores_np)
