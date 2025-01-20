# | filename: script.py
# | code-line-numbers: true

import argparse
import json
import os
import tarfile
from pathlib import Path
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)


def train(
    model_directory,
    train_path,
    validation_path,
    pipeline_path,
    experiment,
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
):
    print(f"Using PyTorch version: {torch.__version__}")

    # Load datasets
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train.pop(X_train.columns[-1]).values
    X_train = X_train.values

    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation.pop(X_validation.columns[-1]).values
    X_validation = X_validation.values

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)

    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    model = SimpleModel(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in validation_loader:
            outputs = model(batch_X)
            predictions = outputs.argmax(dim=1)
            all_preds.extend(predictions.cpu().numpy())
    val_accuracy = accuracy_score(y_validation, all_preds)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Save the model
    model_filepath = Path(model_directory) / "penguins.pth"
    torch.save(model.state_dict(), model_filepath)

    # Save transformation pipelines
    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    # Log metrics and artifacts
    if experiment:
        experiment.log_parameters(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )
        experiment.log_dataset_hash(X_train)
        experiment.log_confusion_matrix(y_validation, np.array(all_preds))
        experiment.log_model("penguins", model_filepath.as_posix())


if __name__ == "__main__":
    # Parse hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args, _ = parser.parse_known_args()

    # Comet experiment
    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None
    )

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", {}))
    job_name = training_env.get("job_name", None) if training_env else None

    if job_name and experiment:
        experiment.set_name(job_name)

    # Train the model
    train(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],
        pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
        experiment=experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
