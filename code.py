#!/usr/bin/env python
# coding: utf-8

# In[722]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import csv

import torch
import captum
from captum.attr import IntegratedGradients
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[723]:


def load_sampled_data(raw_data, sample_fraction=0.01):
    # Extract features and target
    target_col = 'target'
    target = raw_data[target_col]

    # Handle missing target values: either fill with mode or drop rows with NaN in target
    target.fillna(target.mode()[0], inplace=True)  # Filling NaN with the mode of the target
    
    # Initialize StratifiedShuffleSplit to split the data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_fraction, random_state=42)

    # Sample sample_fraction% of data maintaining class distribution
    for train_idx, test_idx in sss.split(raw_data, target):
        sampled_data = raw_data.iloc[test_idx]

    return sampled_data

def create_sequences(data, seq_length, horizon):
    X, X_targets, y, y_targets = [], [], [], []
    feature_names = [col for col in data.columns if col != 'target']  # List of feature column names
    
    for i in range(len(data) - seq_length - horizon + 1):
        # X should contain all columns except 'target' (make sure it's a DataFrame)
        X_seq = data[feature_names].iloc[i:i + seq_length]

        # X_targets should contain only the 'target' column
        X_targets_seq = data['target'].iloc[i:i + seq_length]

        # y should contain the entire row for each sequence, except 'target'
        y_seq = data[feature_names].iloc[i + seq_length + horizon - 1]
        
        # y_targets should contain only the 'target' column
        y_target = data['target'].iloc[i + seq_length + horizon - 1]
        
        # Append sequences to the respective lists
        X.append(X_seq)
        X_targets.append(X_targets_seq)
        y.append(y_seq)
        y_targets.append(y_target)

    return np.array(X), np.array(X_targets), np.array(y), np.array(y_targets)




# **Define LSTM Model**
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


# **Custom Dataset**
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# In[724]:


pca_dim=10
lookforward=5
lookback=60
PCA_DIMENSION = pca_dim

# Model parameters
input_dim = PCA_DIMENSION
hidden_dim = 32
num_layers = 2
output_dim = PCA_DIMENSION
epochs = 500
learning_rate = 0.005
batch_size = 128


# In[725]:


# Load dataset
raw_data = pd.read_csv('old_prometheus_combine_1.csv')

# Define target features to aggregate into a single target column
target_features = [
    'srsdu1_stressType', 'srsdu2_stressType', 'srsdu0_stressType', 'srsdu3_stressType',
    'srscu0_stressType', 'srscu2_stressType', 'srscu3_stressType', 'srscu1_stressType'
]

# Create a unified target column based on the most frequent non-zero stress type
for idx, sample in raw_data.iterrows():
    targets = [sample[target] for target in target_features]
    non_zero_targets = [value for value in targets if value != 0]
    raw_data.at[idx, 'target'] = max(set(non_zero_targets), key=non_zero_targets.count) if non_zero_targets else 0


for idx, sample in raw_data.iterrows():
    for itr in ['key']:  # Loop through the list of columns you want to modify
        val = str(sample[itr])
        
        # Remove the substring 'bsr' from the value
        val = val.replace('bsr', '')
        
        # Convert the modified value to an integer, if possible
        try:
            raw_data.at[idx, itr] = int(val)
        except ValueError:
            # Handle the case where the value cannot be converted to an integer
            raw_data.at[idx, itr] = 0  # Or set it to some default value

# Drop original target feature columns
raw_data = raw_data.drop(columns=target_features)

# raw_data = load_sampled_data(raw_data, sample_fraction=0.01)





# raw_data = np.random.random((1000, 100))
# raw_data = pd.DataFrame(raw_data, columns=[f'feature_{i}' for i in range(100)])
# raw_data['target'] = np.random.randint(0, 4, size=(1000,))
# raw_data['Timestamp'] = pd.date_range(start='1/1/2020', periods=1000, freq='h')
# raw_data['Timestamp'] = raw_data['Timestamp'].astype(str)
# raw_data['Timestamp'] = pd.to_datetime(raw_data['Timestamp'])
# raw_data = raw_data.set_index('Timestamp')



# print(raw_data.head(), raw_data['target'].head())


# In[726]:


# raw_data_2 = pd.read_csv('processed_data_0.csv')
# raw_data_combined = pd.concat([raw_data, raw_data_2], ignore_index=True)
    
# print(f"Number of columns in dataset: {len(raw_data_combined.columns)}")
# print(f"Number of rows in dataset: {len(raw_data_combined)}")


dataset = raw_data.copy()


feature_names = dataset.columns
timestamps = dataset.index
dataset = dataset.drop(columns=['Timestamp'], errors='ignore')

# **Data Preprocessing**
# Handle missing values
dataset = dataset.apply(lambda x: x.fillna(0) if x.isna().all() else x)
threshold = 0.6 * len(dataset)
for col in dataset.columns:
    if dataset[col].isna().sum() > threshold:
        mode_value = dataset[col].mode().iloc[0] if not dataset[col].mode().empty else 0
        dataset.fillna({col: mode_value}, inplace=True)
#            dataset[col].fillna(mode_value, inplace=True)
# dataset = dataset.dropna(subset=['target'])
numeric_cols = dataset.select_dtypes(include=[np.number]).columns
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

# Filter out samples where the target is not in {0, 1, 2, 3}
dataset = dataset[dataset['target'].isin([0, 1, 2, 3])]




# Separate features and target
target_col = 'target'
target = dataset[target_col]
original_features = dataset.drop(columns=['Timestamp', target_col], errors='ignore')


# Binarize the target column
target = dataset[target_col].apply(lambda x: 0 if x == 0 else 1)

# print(f"Number of columns in dataset: {len(original_features.columns)}")


# In[727]:


# Scale features and apply PCA
scaler = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler.fit_transform(original_features)
pca = PCA(n_components=PCA_DIMENSION)
features_pca = pca.fit_transform(features_scaled)

# print(f"Number of columns in dataset: {len(features.columns)}")
# print(f"Number of rows in dataset: {len(features)}")

# print(f"Number of columns in reduced dataset: {len(features_pca[0])}")
# print(f"Number of rows in reduced dataset: {len(features_pca)}")

features_pca = pd.DataFrame(features_pca, columns=[f'pca_{i}' for i in range(PCA_DIMENSION)])

#combine the target column with the pca features
for i in range(len(features_pca)):
    features_pca.at[i, 'target'] = target[i]

# print(features_pca.head())


# In[728]:


# **Prepare Sequences**
# Combine the target column with the features
original_features['target'] = target

X_features, X_targets, y_features, y_targets = create_sequences(features_pca, lookback, lookforward)

# print(f"X_features shape: {X_features.shape}")
# print(f"y_features shape: {y_features.shape}")
# print(f"y_targets shape: {y_targets.shape}")

X_original_features, X_targets, y_original_features, y_targets = create_sequences(original_features, lookback, lookforward)


# print(f"X_original_features shape: {X_original_features.shape}")
# print(f"y_original_features shape: {y_original_features.shape}")
# print(f"y_targets shape: {y_targets.shape}")


# In[729]:


# **Stratified K-Fold Cross-Validation**
n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize metric accumulators
accuracy_scores_actual = []
f1_scores_actual = []
classification_reports_actual = []
accuracy_scores_forecasted = []
f1_scores_forecasted = []
classification_reports_forecasted = []
total_conf_matrix_actual = None
total_conf_matrix_forecasted = None
RMSE_PCA = []

print("Unique values in target:", target.unique())
print("\nOverall class distribution:", dict(zip(*np.unique(y_targets, return_counts=True))))


# In[730]:


class SequentialGenerator(nn.Module):
    def __init__(self, latent_dim, sequence_length, hidden_dim, output_dim):
        super(SequentialGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # LSTM to process the sequence data
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # Linear layer to map LSTM output to the desired output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, previous_sequence):
        # z shape: (batch_size, latent_dim)
        # previous_sequence shape: (batch_size, sequence_length)
        
        # Repeat z across the sequence length dimension
        z_repeated = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # LSTM to generate new sequence
        lstm_out, _ = self.lstm(z_repeated)
        
        # Output layer to generate the next time step
        generated_output = self.fc(lstm_out[:, -1, :])
        
        return generated_output

class SequentialDiscriminator(nn.Module):
    def __init__(self, sequence_length, hidden_dim, input_size):  # Changed parameter name
        super(SequentialDiscriminator, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Match input_size to actual feature dimension (100)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)  # lstm_out shape: [batch_size, hidden_size]
        return self.sigmoid(self.fc(lstm_out))


# In[731]:


# **Cross-Validation Loop**
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_features, y_targets), start=1):
    # Split data into train and test folds
    X_train_fold, X_test_fold = X_features[train_idx], X_features[test_idx]
    y_train_fold, y_test_fold = y_features[train_idx], y_features[test_idx]

    X_original_train_fold, X_original_test_fold = X_original_features[train_idx], X_original_features[test_idx]
    y_original_train_fold, y_original_test_fold =y_original_features[train_idx], y_original_features[test_idx]
    
    y_train_target, y_test_target = y_targets[train_idx], y_targets[test_idx]
    
    
    # Train LSTM model
    train_dataset = SequenceDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    lstm_model = LSTMForecaster(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    if(not os.path.exists('models')):
        os.makedirs('models')
    if os.path.exists(f'models/lstm_model_fold_{fold_idx}.pt'):
        lstm_model.load_state_dict(torch.load(f"models/lstm_model_fold_{fold_idx}.pt"))
        lstm_model.eval()
    else:
        for epoch in range(epochs):
            lstm_model.train()
            epoch_loss = 0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(device, dtype=torch.float32)
                batch_targets = batch_targets.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                predictions = lstm_model(batch_sequences)
                d_loss = criterion(predictions, batch_targets)
                predictions_inverse = pca.inverse_transform(predictions.cpu().detach().numpy())
                predictions_inverse_tensor = torch.tensor(predictions_inverse, dtype=torch.float32).to(device)
                batch_inverse_targets = pca.inverse_transform(batch_targets.cpu().detach().numpy())
                batch_inverse_targets_tensor = torch.tensor(batch_inverse_targets, dtype=torch.float32).to(device)
                o_loss = criterion(predictions_inverse_tensor, batch_inverse_targets_tensor)
                final_loss = d_loss + o_loss / 30
                final_loss.backward()
                optimizer.step()
                epoch_loss += final_loss.item()
        torch.save(lstm_model.state_dict(), f"models/lstm_model_fold_{fold_idx}.pt")
    
    # Forecasting with LSTM
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
        y_pred_test = lstm_model(X_test_tensor)
        y_pred_pca_inverse = pca.inverse_transform(y_pred_test.cpu().numpy())
        y_true_pca_inverse = pca.inverse_transform(y_test_fold)
        rmse_pca = np.sqrt(np.mean((y_true_pca_inverse - y_pred_pca_inverse) ** 2))
        RMSE_PCA.append(rmse_pca)
        y_pred_test_original = scaler.inverse_transform(y_pred_pca_inverse)


    JVGAN_features = []
    # # Filter normal data where target == 0
    # original_features['Timestamp'] =  timestamps

    JVGAN_features = original_features[original_features['target'] == 0]
    
    # Ensure the data is sorted by timestamp if it's not already
    JVGAN_features = JVGAN_features.sort_values(by='Timestamp')

    # Drop the target column as it is not needed for JVGAN training
    JVGAN_features = JVGAN_features.drop(columns=['target'])

    # anomaly detection using JVGAN
    JVGAN_features = pd.DataFrame(JVGAN_features)
    # print(f"JVGAN Features:\n{JVGAN_features.head()}")
    
    # Scale features and apply PCA
    JVGAN_scaler = MinMaxScaler(feature_range=(-1, 1))
    JVGAN_features_scaled = scaler.fit_transform(JVGAN_features)

    # Train JVGAN
    real_sequences = torch.tensor(JVGAN_features_scaled, dtype=torch.float32).to(device)

    # Hyperparameters
    JVGAN_latent_dim = 100  # Latent dimension for noise vector
    JVGAN_sequence_length = lookback  # Number of previous time steps to condition on
    JVGAN_LSTM_hidden_dim = 128  # Hidden dimension for LSTM layers
    JVGAN_output_dim = JVGAN_features_scaled.shape[1]


    # Initialize models
    generator = SequentialGenerator(JVGAN_latent_dim, JVGAN_sequence_length, JVGAN_LSTM_hidden_dim, JVGAN_output_dim).to(device)
    # When creating discriminator:
    discriminator = SequentialDiscriminator(
        sequence_length=JVGAN_sequence_length,
        hidden_dim=JVGAN_LSTM_hidden_dim,
        input_size=JVGAN_features_scaled.shape[1]  # Should be 100 for your data
    ).to(device)    

    # Loss and optimizers
    criterion = nn.BCELoss()  # Binary cross-entropy for GAN
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 200
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(real_sequences) - JVGAN_sequence_length, batch_size):
            batch = real_sequences[i:i+batch_size]
            
            # Train Discriminator
            optimizer_d.zero_grad()
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            real_output = discriminator(batch)
            d_loss_real = criterion(real_output, real_labels)
            
            z = torch.randn(batch_size, JVGAN_latent_dim).to(device)
            fake_sequence = generator(z, batch)
            fake_output = discriminator(fake_sequence.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            
            fake_output = discriminator(fake_sequence)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            optimizer_g.step()
        if(epoch % 10 == 0):
            print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


    # **Classification on Actual Future Features**
    # Anomaly Detection using Discriminator
    # with torch.no_grad():
    #     test_scores = discriminator(X_test_tensor).numpy().flatten()
    #     anomaly_preds = (test_scores < 0.5).astype(int)  # Lower scores indicate anomalies

    with torch.no_grad():
        # print(f"X_test_fold shape: {X_original_test_fold.shape}")
        # X_test_tensor is 3D: (batch_size, seq_length, input_dim)
        JVGAN_X_test_tensor = torch.tensor(X_original_test_fold, dtype=torch.float32).to(device)
        
        # Select the last timestep
        JVGAN_X_last_timestep = JVGAN_X_test_tensor[:, -1, :]  # Shape: (batch_size, input_dim)

        # Pass the last timestep to the discriminator
        # print(f"X_last_timestep shape: {JVGAN_X_last_timestep.shape}")
        test_scores = discriminator(JVGAN_X_last_timestep).cpu().numpy().flatten()
        anomaly_preds = (test_scores < 0.5).astype(int)  # Lower scores indicate anomalies
    
    torch.cuda.empty_cache()
    lstm_model = LSTMForecaster(input_dim, hidden_dim, num_layers, output_dim).to(device)
    lstm_model.load_state_dict(torch.load(f"models/lstm_model_fold_{fold_idx}.pt", map_location=device, weights_only=False))
    lstm_model.eval()

    # **Classification on Actual Future Features**
    print(f"\nClassification Report for Actual Future Features (Fold {fold_idx}):")
    print(classification_report(y_test_target, anomaly_preds))
    conf_matrix_actual = confusion_matrix(y_test_target, anomaly_preds)
    print(f"Confusion Matrix for Actual Features (Fold {fold_idx}):")
    print(conf_matrix_actual)
    accuracy_actual = accuracy_score(y_test_target, anomaly_preds)
    f1_actual = f1_score(y_test_target, anomaly_preds, average='weighted')
    accuracy_scores_actual.append(accuracy_actual)
    f1_scores_actual.append(f1_actual)
    classification_reports_actual.append(classification_report(y_test_target, anomaly_preds, output_dict=True))

    # **Classification on Forecasted Features**
    y_jvgan_pred_forecasted = discriminator(torch.tensor(y_pred_test_original, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
    y_jvgan_pred_forecasted = (y_jvgan_pred_forecasted < 0.5).astype(int)  # Lower scores indicate anomalies
    print(f"\nClassification Report for Forecasted Features (Fold {fold_idx}):")
    print(classification_report(y_test_target, y_jvgan_pred_forecasted))
    conf_matri_forecasted = confusion_matrix(y_test_target, y_jvgan_pred_forecasted)
    print(f"Confusion Matrix for Forecasted Features (Fold {fold_idx}):")
    print(conf_matrix_forecasted)
    accuracy_forecasted = accuracy_score(y_test_target, y_jvgan_pred_forecasted)
    f1_forecasted = f1_score(y_test_target, y_jvgan_pred_forecasted, average='weighted')
    accuracy_scores_forecasted.append(accuracy_forecasted)
    f1_scores_forecasted.append(f1_forecasted)
    classification_reports_forecasted.append(classification_report(y_test_target, y_jvgan_pred_forecasted, output_dict=True))
            
    for i in range(len(anomaly_preds)):
        if anomaly_preds[i] != 0:
            # here add the code for the integrated gradients
            # Initialize the IntegratedGradients object
            ig = IntegratedGradients(discriminator) # discriminator is the model used for anomaly detection
            # Get the input tensor
            input_tensor = torch.tensor(y_pred_test_original, dtype=torch.float32).to(device)
            # Get the baseline tensor
            baseline_tensor = torch.zeros_like(input_tensor)
            # Get the attributions
            attributions, delta = ig.attribute(input_tensor, baseline_tensor, target=0, return_convergence_delta=True)
            
            
            # Get the attributions as numpy array
            attributions = attributions.cpu().detach().numpy()
            # Get the delta as numpy array
            delta = delta.cpu().detach().numpy()

            if not os.path.exists('RCA.csv'):
                with open('RCA.csv', mode='w', newline='') as f:
                    writer = csv.writer(f)
                    # create a dictionary with feature_names and write attributions sequentially
                    attributions_dict = dict(zip(feature_names, attributions[i]))
                    attributions_dict['predicted_target'] = anomaly_preds[i]
                    writer.writerow(attributions_dict.keys())  # Write the column names (keys)

            with open('RCA.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                # create a dictionary with feature_names and write attributions sequentially
                attributions_dict = dict(zip(feature_names, attributions[i]))
                attributions_dict['predicted_target'] = anomaly_preds[i]
                writer.writerow(attributions_dict.values())

    # Accumulate confusion matrices
    if total_conf_matrix_actual is None:
        total_conf_matrix_actual = conf_matrix_actual
    else:
        total_conf_matrix_actual += conf_matrix_actual

    if total_conf_matrix_forecasted is None:
        total_conf_matrix_forecasted = conf_matrix_forecasted
    else:
        total_conf_matrix_forecasted += conf_matrix_forecasted

    torch.cuda.empty_cache()


# In[ ]:


# **Compute Average Metrics**
avg_accuracy_actual = np.mean(accuracy_scores_actual)
avg_f1_actual = np.mean(f1_scores_actual)
avg_accuracy_forecasted = np.mean(accuracy_scores_forecasted)
avg_f1_forecasted = np.mean(f1_scores_forecasted)
avg_rmse_pca = np.mean(RMSE_PCA)
avg_conf_matrix_actual = total_conf_matrix_actual / n_splits
avg_conf_matrix_forecasted = total_conf_matrix_forecasted / n_splits

# Average classification report for actual features
avg_report_actual = {}
for key in classification_reports_actual[0].keys():
    if key not in ['accuracy', 'macro avg', 'weighted avg']:
        avg_report_actual[key] = {metric: np.mean([r[key][metric] for r in classification_reports_actual])
                                    for metric in ['precision', 'recall', 'f1-score']}
avg_report_actual['macro avg'] = {metric: np.mean([r['macro avg'][metric] for r in classification_reports_actual])
                                    for metric in ['precision', 'recall', 'f1-score']}
avg_report_actual['weighted avg'] = {metric: np.mean([r['weighted avg'][metric] for r in classification_reports_actual])
                                    for metric in ['precision', 'recall', 'f1-score']}

# Average classification report for forecasted features
avg_report_forecasted = {}
for key in classification_reports_forecasted[0].keys():
    if key not in ['accuracy', 'macro avg', 'weighted avg']:
        avg_report_forecasted[key] = {metric: np.mean([r[key][metric] for r in classification_reports_forecasted])
                                        for metric in ['precision', 'recall', 'f1-score']}
avg_report_forecasted['macro avg'] = {metric: np.mean([r['macro avg'][metric] for r in classification_reports_forecasted])
                                        for metric in ['precision', 'recall', 'f1-score']}
avg_report_forecasted['weighted avg'] = {metric: np.mean([r['weighted avg'][metric] for r in classification_reports_forecasted])
                                        for metric in ['precision', 'recall', 'f1-score']}

# **Display Results**
print("\n\nAverage Metrics for Classification on Actual Future Features:")
print(f"Average Accuracy: {avg_accuracy_actual}")
print(f"Average F1-Score: {avg_f1_actual}")
print("Average Classification Report:")
for key, metrics in avg_report_actual.items():
    if key not in ['macro avg', 'weighted avg']:
        print(f"Class {key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    else:
        print(f"{key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
print("Average Confusion Matrix (Actual Features):")
print(avg_conf_matrix_actual)

print("\n\nAverage Metrics for Classification on Forecasted Features:")
print(f"Average Accuracy: {avg_accuracy_forecasted}")
print(f"Average F1-Score: {avg_f1_forecasted}")
print("Average Classification Report:")
for key, metrics in avg_report_forecasted.items():
    if key not in ['macro avg', 'weighted avg']:
        print(f"Class {key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    else:
        print(f"{key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
print(f"\nAverage RMSE (PCA Inverse): {avg_rmse_pca}")
print("Average Confusion Matrix (Forecasted Features):")
print(avg_conf_matrix_forecasted)


# In[ ]:
