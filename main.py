import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from skin_model import SkinDataset, SkinModel, Trainer, Evaluator

def main(activation_fn, output_fn, sampler, layer1, layer2, epochs, batch_size, learning_rate, dataset_filename, test_images_folder, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkinDataset(dataset_filename)

    model = SkinModel(layer1=layer1, layer2=layer2, activation=activation_fn, output=output_fn, classes=1)

    loss_fn = torch.nn.BCELoss() if output_fn == "sigmoid" else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if sampler:
        class_weights = [1.0, 1.0]
        sample_weights = torch.tensor([class_weights[int(label)] for _, label in dataset], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    trainer = Trainer(model=model, dataset=dataset, epochs=epochs, batch=batch_size, 
                      device=device, loss_fn=loss_fn, optimizer=optimizer,
                      activation=activation_fn, output=output_fn, sampler=sampler, threshold=threshold)

    print("Iniciando el entrenamiento...")
    trainer.train()

    evaluator = Evaluator(model=model, device=device, threshold=threshold)
    
    # Llamar al nuevo método loss_tra_val para mostrar el gráfico de pérdida
    evaluator.loss_tra_val(trainer.losses, trainer.val_losses)
    
    print("Evaluando el modelo...")
    evaluator.evaluate_images_with_masks(test_images_folder)

    print("Evaluación completada.")

main(
    activation_fn="relu",
    output_fn="sigmoid",
    sampler=True,
    layer1=8,
    layer2=16,
    epochs=10,
    batch_size=32,
    learning_rate=0.01,
    dataset_filename="skin_nskin.npy",
    test_images_folder="./dataset_with_mask",
    threshold=0.5
)
