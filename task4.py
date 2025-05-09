import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from datetime import datetime

from task1 import task1


def transform_to_gaf(data_2d, image_size=24):
    transformer = GramianAngularField(image_size=image_size, method='summation')
    return transformer.transform(data_2d)


def apply_gaf_transformation(df):
    df_gaf = df.copy()
    features = ['load', 'wind', 'solar']

    for feature in features:
        data = np.vstack(df[f'{feature}_24h'])
        gaf_images = transform_to_gaf(data)
        df_gaf[f'{feature}_gaf'] = [img for img in gaf_images]

    return df_gaf


# Dataset и DataLoader
class EnergyDataset(Dataset):
    def __init__(self, df, labels):
        self.load_images = np.stack(df['load_gaf'])
        self.wind_images = np.stack(df['wind_gaf'])
        self.solar_images = np.stack(df['solar_gaf'])
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.stack([
            self.load_images[idx],
            self.wind_images[idx],
            self.solar_images[idx]
        ], axis=0)
        return torch.FloatTensor(image), torch.LongTensor([self.labels[idx]])


# Модель 2D-CNN
class SeasonClassifier2DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SeasonClassifier2DCNN, self).__init__()
        # Input shape: (batch_size, 3, 24, 24)
        # Output shape: (batch_size, 16, 24, 24) [padding=1 preserves spatial dims]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # Shape preserved: (batch_size, 16, 24, 24)

        # Input: (batch_size, 16, 24, 24)
        # Output: (batch_size, 16, 12, 12) [kernel_size=2 halves the dimensions]
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Input: (batch_size, 16, 12, 12)
        # Output: (batch_size, 32, 12, 12) [padding=1 preserves spatial dims]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # Shape preserved: (batch_size, 32, 12, 12)

        # Input: (batch_size, 32, 12, 12)
        # Output: (batch_size, 32, 6, 6) [kernel_size=2 halves the dimensions]
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Flatten occurs here in forward(): (batch_size, 32, 6, 6) -> (batch_size, 32*6*6=1152)

        # Input: (batch_size, 1152)
        # Output: (batch_size, 128)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.relu3 = nn.ReLU()  # Shape preserved: (batch_size, 128)

        # Input: (batch_size, 128)
        # Output: (batch_size, num_classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Функция обучения
# Функция обучения
def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []  # Добавлено для хранения train accuracy
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0  # Добавлено для подсчета правильных предсказаний
        train_total = 0  # Добавлено для общего количества примеров

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Вычисляем accuracy на тренировочном батче
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.squeeze()).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total  # Вычисляем train accuracy

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)  # Сохраняем train accuracy

        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.squeeze()).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, '  # Добавлен вывод train accuracy
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}')

    return train_losses, train_accuracies, val_losses, val_accuracies  # Обновлен возвращаемый результат


# Основной процесс выполнения
def main():
    # 1. Загрузка и подготовка данных
    train_df, val_df, test_df = task1()

    # 2. Преобразование в GAF изображения
    train_gaf = apply_gaf_transformation(train_df)
    val_gaf = apply_gaf_transformation(val_df)
    test_gaf = apply_gaf_transformation(test_df)

    # 3. Подготовка DataLoader
    le = LabelEncoder()
    train_labels = le.fit_transform(train_gaf['season'])
    val_labels = le.transform(val_gaf['season'])
    test_labels = le.transform(test_gaf['season'])

    batch_size = 32
    train_dataset = EnergyDataset(train_gaf, train_labels)
    val_dataset = EnergyDataset(val_gaf, val_labels)
    test_dataset = EnergyDataset(test_gaf, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 4. Создание и обучение модели
    model = SeasonClassifier2DCNN()
    train_losses, val_losses, val_accuracies, d = train_model(model, train_loader, val_loader)

    # 5. Оценка на тестовом наборе
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

            # Сохраняем предсказания и истинные метки для отчета
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Classification Report
    class_names = ['winter', 'spring', 'summer', 'autumn']
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))


if __name__ == '__main__':
    main()