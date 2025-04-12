import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from skin_model import SkinDataset, SkinModel, Trainer, Evaluator
import os
import sys




def main(activation_fn, output_fn, sampler_enabled, layer1, layer2, epochs, batch_size,
         learning_rate, weight_decay, dataset_filename, test_images_folder, threshold=0.5):
    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print current directory for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Loading dataset from: {dataset_filename}")

    # Load and prepare dataset
    dataset = SkinDataset(dataset_filename)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Create model
    classes = 2 if output_fn == "softmax" else 1
    model = SkinModel(layer1=layer1, layer2=layer2,
                      activation=activation_fn, output=output_fn, classes=classes)
    print(
        f"Model created with architecture: {layer1}-{layer2}-{layer2//2}-{classes}")

    # Define loss function
    if output_fn == "sigmoid":
        loss_fn = torch.nn.BCELoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Use Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set up weighted sampler if enabled to handle class imbalance
    if sampler_enabled:
        print("Using weighted sampler to handle class imbalance")
        labels = [int(label.item()) for _, label in dataset]
        class_sample_count = np.bincount(labels)
        print(f"Class distribution: {class_sample_count}")
        class_weights = 1. / class_sample_count
        sample_weights = [class_weights[label] for label in labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        sampler = None

    # Create and run trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch=batch_size,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        activation=activation_fn,
        output=output_fn,
        sampler=sampler,
        threshold=threshold
    )

    print(
        f"Starting training for {epochs} epochs with batch size {batch_size}")
    trainer.train()

    # Evaluate on test images
    print(f"Evaluating model on test images from {test_images_folder}")
    evaluator = Evaluator(model=model, device=device, threshold=threshold)
    evaluator.evaluate_images_with_masks(test_images_folder)


# Run with improved hyperparameters
main(
    activation_fn="relu",  # ReLU activation for better gradient flow
    output_fn="sigmoid",   # Sigmoid for binary classification
    sampler_enabled=True,  # Enable class balancing
    layer1=8,             # Larger hidden layers for better capacity
    layer2=16,            # Increased from 16 to 128
    epochs=20,             # More training epochs
    batch_size=32,         # Increased batch size for better batch normalization
    learning_rate=0.001,   # Lower learning rate for Adam optimizer
    weight_decay=1e-4,     # L2 regularization to prevent overfitting
    dataset_filename="skin_nskin.npy",
    test_images_folder=r"C:\Users\alvar\Documents\Universidad\tarea1\tarea1\dataset_with_mask",
    threshold=0.5          # Balanced threshold for binary classification
)
