import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.metrics import accuracy_score, classification_report

from task1 import task1


class MLP(nn.Module):
    def __init__(self, input_size=72, hidden_size=64, output_size=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def prepare_data(df):
    features = np.array([np.concatenate([day['load_24h'],
                                         day['wind_24h'],
                                         day['solar_24h']])
                         for day in df.to_dict('records')])
    season_to_idx = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    labels = np.array([season_to_idx[day['season']] for day in df.to_dict('records')])
    return torch.FloatTensor(features), torch.LongTensor(labels)


def train_model(train_df, val_df, epochs=50, batch_size=32):
    # Подготовка данных
    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)

    # Создаем DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели
    model = MLP(input_size=72, hidden_size=64, output_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Цикл обучения
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

        # Сохраняем метрики
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


def evaluate_model(model, test_df):
    X_test, y_test = prepare_data(test_df)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f'Test Accuracy: {accuracy:.4f}')

    print("\nTest Set Predictions:")
    print(predicted.numpy())


    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), predicted.numpy(),
                                target_names=['winter', 'spring', 'summer', 'autumn']))
    return accuracy


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def task2(train_df_scaled, val_df_scaled, test_df_scaled):
    print("\n=== Task 2: MLP Implementation ===")

    # Обучение модели
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        train_df_scaled, val_df_scaled, epochs=50)

    # Визуализация метрик
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Оценка на тестовых данных
    test_acc = evaluate_model(model, test_df_scaled)
    return model

train_df_scaled, val_df_scaled, test_df_scaled = task1()
task2(train_df_scaled, val_df_scaled, test_df_scaled)
