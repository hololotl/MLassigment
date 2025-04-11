import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

from task1 import task1


class CNN1D(nn.Module):
    def __init__(self, input_channels=3, seq_len=24, num_classes=4):
        super(CNN1D, self).__init__()

        # (batch_size, 3, 24) -> (batch_size, 32, 22)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding='valid')
        self.relu = nn.ReLU()

        # (batch_size, 32, 22) -> (batch_size, 32, 11)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # (batch_size, 32 * 11) -> (batch_size, 64)
        self.fc1 = nn.Linear(32 * 11, 64)

        # (batch_size, 64) -> (batch_size, 4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def prepare_cnn_data(df):
    features = np.array([np.stack([day['load_24h'],
                                   day['wind_24h'],
                                   day['solar_24h']], axis=0)
                         for day in df.to_dict('records')])

    season_to_idx = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    labels = np.array([season_to_idx[day['season']] for day in df.to_dict('records')])

    return torch.FloatTensor(features), torch.LongTensor(labels)


def train_cnn(train_df, val_df, epochs=50, batch_size=32):
    X_train, y_train = prepare_cnn_data(train_df)
    X_val, y_val = prepare_cnn_data(val_df)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CNN1D(input_channels=3, seq_len=24, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = accuracy_score(y_val.numpy(), val_predicted.numpy())

        # Логирование
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_cnn(model, test_df):
    X_test, y_test = prepare_cnn_data(test_df)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f'Test Accuracy: {accuracy:.4f}')

        print("\nTest Set Predictions:")
        print(predicted.numpy())

        print("\nClassification Report:")
        print(classification_report(y_test.numpy(), predicted.numpy(),
                                    target_names=['winter', 'spring', 'summer', 'autumn']))

    return accuracy


def task3(train_df_scaled, val_df_scaled, test_df_scaled):
    print("\n=== Task 3: 1D-CNN Implementation ===")

    model, train_losses, val_losses, train_accs, val_accs = train_cnn(
        train_df_scaled, val_df_scaled, epochs=50)

    test_acc = evaluate_cnn(model, test_df_scaled)
    return model

train_df_scaled, val_df_scaled, test_df_scaled = task1()
task3(train_df_scaled, val_df_scaled, test_df_scaled)