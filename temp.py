import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data from CSV
df = pd.read_csv("Raw_data_1Day_2022_site_301_Anand_Vihar_Delhi_DPCC_1Day.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert the Timestamp column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set the Timestamp column as the index
df.set_index('Timestamp', inplace=True)

# Resample the data to a daily frequency, taking the mean of each day
df_daily = df.resample('D').mean()

# Ensure that the resampled dataset is not empty
if df_daily.empty:
    print("Resampled dataset is empty. Check your data or resampling frequency.")
    exit()

# Split the data into features (X) and target (y) variables
X = df_daily.drop(columns=['PM2.5 (µg/m³)'])  # Features excluding the target variable (PM2.5)
y = df_daily['PM2.5 (µg/m³)']  # Target variable (PM2.5)

# Check if the dataset contains sufficient samples for splitting
if len(X) == 0 or len(y) == 0:
    print("Dataset is empty after preprocessing. Check your data preprocessing steps.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize/Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a custom dataset class for time-series data
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Initialize the model architecture
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size):
        super(TimeSeriesModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the dataset and DataLoader
train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the model, loss function, and optimizer
model = TimeSeriesModel(input_size=X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_model.pth")
