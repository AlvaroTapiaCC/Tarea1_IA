import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from skin_model import SkinDataset, SkinModel, Trainer, Evaluator

def main(activation_fn, output_fn, sampler_enabled, layer1, layer2, epochs, batch_size, learning_rate, dataset_filename, test_images_folder, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkinDataset(dataset_filename)

    classes = 2 if output_fn == "softmax" else 1
    model = SkinModel(layer1=layer1, layer2=layer2, activation=activation_fn, output=output_fn, classes=classes)

    if output_fn == "sigmoid":
        loss_fn = torch.nn.BCELoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if sampler_enabled:
        labels = [int(label.item()) for _, label in dataset]
        class_sample_count = np.bincount(labels)
        class_weights = 1. / class_sample_count
        sample_weights = [class_weights[label] for label in labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        sampler = None

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

    trainer.train()

    evaluator = Evaluator(model=model, device=device, threshold=threshold)
    evaluator.evaluate_images_with_masks(test_images_folder)

main(
    activation_fn="relu",
    output_fn="sigmoid",
    sampler_enabled=True,
    layer1=8,
    layer2=16,
    epochs=10,
    batch_size=16,
    learning_rate=0.01,
    dataset_filename="skin_nskin.npy",
    test_images_folder=r"C:\Users\alvar\Documents\Universidad\2025-1\Artificial Intelligence\Tarea1\dataset_with_mask",
    threshold=0.2
)
