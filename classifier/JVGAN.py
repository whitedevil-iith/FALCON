#!/usr/bin/env python
# coding: utf-8

# In[418]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[419]:


# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_features, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Adjust the first layer to match the input dimensions
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 512),  # Adjust this to match input dimensions
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_features),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # Concatenate latent vector and labels
        inputs = torch.cat([z, labels], dim=1)
        return self.model(inputs)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Discriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Linear(num_features + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Concatenate features and labels
        inputs = torch.cat([x, labels], dim=1)
        return self.model(inputs)


# In[420]:


# Load dataset
dataset = pd.read_csv('old_prometheus_combine_1.csv')
dataset = dataset[:int(0.01*len(dataset))]

dataset.index = dataset['Timestamp']
dataset = dataset.drop(columns=['Timestamp'])
print(dataset.head())


# # dataset.shape

# import pandas as pd
# import numpy as np

# # Set random seed for reproducibility
# np.random.seed(42)

# # Create dummy dataset with 4 features and 2 targets
# num_samples = 100
# data = {
#     'host': np.random.randn(num_samples),
#     'srscu0': np.random.randn(num_samples),
#     'srscu1': np.random.randn(num_samples),
#     'srscu2': np.random.randn(num_samples),
#     'srscu3': np.random.randn(num_samples),
#     'srsdu0': np.random.randn(num_samples),
#     'srsdu1': np.random.randn(num_samples),
#     'srsdu2': np.random.randn(num_samples),
#     'srsdu3': np.random.randn(num_samples),
#     'srscu0_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srscu1_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srscu2_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srscu3_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srsdu0_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srsdu1_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srsdu2_stressType': np.random.randint(0, 5, num_samples),  # Binary target
#     'srsdu3_stressType': np.random.randint(0, 5, num_samples),  # Binary target

# }


# # Create a DataFrame
# dataset = pd.DataFrame(data)

# # Create a timestamp index (assuming the start date is '2025-04-01')
# dataset.index = pd.date_range(start='2025-04-01', periods=num_samples, freq='D')


# # Show the first few rows of the dataset
# print(dataset.head())


# In[421]:


for idx, sample in dataset.iterrows():
    if('key' in dataset.columns):
        for itr in ['key']:  # Loop through the list of columns you want to modify
            val = str(sample[itr])
            
            # Remove the substring 'bsr' from the value
            val = val.replace('bsr', '')
            
            # Convert the modified value to an integer, if possible
            try:
                dataset.at[idx, itr] = int(val)
            except ValueError:
                # Handle the case where the value cannot be converted to an integer
                dataset.at[idx, itr] = 0  # Or set it to some default value


# In[422]:


NoOfCUs = 4
NoOfDUs = 4

# Creating Topology
topology = {}

# Form the graph where srscu0 connects to srsdu0, srscu1 to srsdu1, and so on
for i in range(min(NoOfCUs, NoOfDUs)):  # Prevent index errors
    topology[f"srscu{i}"] = [f"srsdu{i}"]

# Display the graph
print(topology)


# In[423]:


common_features = dataset.columns.tolist()
container_specific_features = {}

# Loop through and remove columns containing specific substrings
for i in range(NoOfDUs+1):
    common_features = [col for col in common_features if f"srscu{i}" not in col and f"srsdu{i}" not in col]

# Store container-specific dataframes instead of lists
for i in range(NoOfCUs+1):
    container_specific_features[f'srscu{i}'] = dataset[[col for col in dataset.columns.tolist() if f"srscu{i}" in col]]

for i in range(NoOfDUs+1):
    container_specific_features[f'srsdu{i}'] = dataset[[col for col in dataset.columns.tolist() if f"srsdu{i}" in col]]

# # Print the remaining features
# print(len(common_features), common_features)

# print("Before:")

# # Print container-specific features (as dataframes now)
# for i in range(NoOfCUs):
#     print(f"srscu{i}:")
#     print(container_specific_features[f'srscu{i}'].shape)
#     print(container_specific_features[f'srscu{i}'].head())

# for i in range(NoOfDUs):
#     print(f"srsdu{i}:")
#     print(container_specific_features[f'srsdu{i}'].shape)
#     print(container_specific_features[f'srsdu{i}'].head())

# Filter out columns containing 'stepStress' from the container-specific dataframes
for i in range(NoOfCUs):
    container_specific_features[f'srscu{i}'] = container_specific_features[f'srscu{i}'].loc[:, ~container_specific_features[f'srscu{i}'].columns.str.contains('stepStress')]

for i in range(NoOfDUs):
    container_specific_features[f'srsdu{i}'] = container_specific_features[f'srsdu{i}'].loc[:, ~container_specific_features[f'srsdu{i}'].columns.str.contains('stepStress')]

# print("After:")

# # Print container-specific features (after filtering)
# for i in range(NoOfCUs):
#     print(f"srscu{i}:")
#     print(container_specific_features[f'srscu{i}'].shape)
#     print(container_specific_features[f'srscu{i}'].head())

# for i in range(NoOfDUs):
#     print(f"srsdu{i}:")
#     print(container_specific_features[f'srsdu{i}'].shape)
#     print(container_specific_features[f'srsdu{i}'].head())


# In[424]:


# Iterate through the topology and combine features
combined_samples = {}

for CU in topology.keys():
    # The CU container-specific features
    CU_features = container_specific_features[CU]
    
    # The connected DUs (from topology)
    connected_DUs = topology[CU]
    
    # Add CU-specific features to the combined list
    CU_features_list = CU_features.columns.tolist()
    
    # Extract the CU stress type column (if exists)
    CU_stressType = f'{CU}_stressType' if f'{CU}_stressType' in CU_features.columns else None
    
    # Add DU-specific features to the combined list for each connected DU
    for DU in connected_DUs:
        # Ensure DU exists in container_specific_features
        if DU in container_specific_features:
            DU_features = container_specific_features[DU]
            DU_features_list = DU_features.columns.tolist()

            # Combine CU and DU features (remove the stress type columns from features)
            combined_features = common_features.copy()  # Start with the common features
            
            # Modify these lines:
            combined_features.extend(CU_features_list)  # Keep all CU features
            combined_features.extend(DU_features_list)  # Keep all DU features

            

            # Extract targets and remove them from features
            targets = [col for col in combined_features if '_stressType' in col]

            # To keep stressType columns temporarily:
            combined_samples[(CU, DU)] = {
                'features': list(set(combined_features) - set(targets)),  # Include targets in features temporarily
                'targets': list(set(targets))
            }

        else:
            print(f"Error: {DU} not found in container_specific_features!")
            continue  # Skip this DU if not found in container_specific_features
    
# print(combined_samples)
# # Print the results for each CU-DU pair and its combined features
# for (CU, DU), sample in combined_samples.items():
#     print(f"Host and CU: {CU}, DU: {DU} - Combined Features:")
#     print(f"Number of Features: {len(sample['features'])}")
#     print(f"Number of Targets: {len(sample['targets'])}")
#     print(sample['features'][:10])  # Print first 10 features as a preview
#     print("----" * 10)


# In[ ]:


AUC_ROC = []
PRECISION = []
RECALL = []
F1_SCORE = []

for (CU, DU), sample in combined_samples.items():
    print(f"Training on data of Host, {CU} and {DU}")


    # Load data with duplicate handling
    raw_data = dataset[sample['features'] + sample['targets']].copy()
    raw_data = raw_data.loc[:, ~raw_data.columns.duplicated()]  # KEY FIX


    # Filter using targets
    for target_col in sample['targets']:
        raw_data = raw_data[raw_data[target_col].isin([0, 1, 2, 3])]


    # **Data Preprocessing**
    # Handle missing values
    raw_data = raw_data.apply(lambda x: x.fillna(0) if x.isna().all() else x)


    threshold = 0.6 * len(raw_data)
    raw_data = raw_data.loc[:, ~raw_data.columns.duplicated()]  # Remove duplicates

    for col in raw_data.columns:
        nan_count = raw_data[col].isna().sum()
        if int(nan_count) > threshold:  # Explicit scalar conversion
            mode_value = raw_data[col].mode().iloc[0] if not raw_data[col].mode().empty else 0
            raw_data[col].fillna(mode_value, inplace=True)

    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
    raw_data[numeric_cols] = raw_data[numeric_cols].fillna(raw_data[numeric_cols].mean())


    # **Convert target columns to binary (0 or 1)**
    # Instead of using a loop over rows, we can do it in a vectorized way
    for target_col in sample['targets']:
        raw_data[target_col] = raw_data[target_col].apply(lambda x: 1 if x != 0 else 0)


    # Create unified target column
    raw_data['target'] = 0
    for idx in raw_data.index:
        if any(raw_data.loc[idx, sample['targets']] == 1):
            raw_data.at[idx, 'target'] = 1

    
    X = raw_data.drop(columns=sample['targets']+['target'])
    Y = raw_data['target']

    # To avoid division by zero:
    X = (X - X.mean()) / (X.std() + 1e-8)

    train_idx = int(0.8 * len(X))

    # Concatenate X and Y into raw_data (pd.concat is used to join the features and targets)
    raw_data = pd.concat([X, Y], axis=1)


    raw_data_training = raw_data[:train_idx]
    raw_data_testing = raw_data[train_idx:]


    # Convert all columns to float16
    raw_data_training = raw_data_training.astype(np.float16)
    raw_data_testing = raw_data_testing.astype(np.float16)


    # Convert to PyTorch tensors
    features = torch.FloatTensor(raw_data_training.drop(columns=['target']).values).to(device)
    labels = torch.LongTensor(raw_data_training['target'].values).to(device)

    # Create dataset and dataloader
    trainingDataset = TensorDataset(features, labels)
    trainingDataloader = DataLoader(trainingDataset, batch_size=32, shuffle=True)


    # Hyperparameters
    latent_dim = raw_data_training.shape[1] - 1
    # latent_dim = 100
    num_features = raw_data_training.shape[1] - 1
    num_classes = 2
    lr = 0.0002
    num_epochs = 100

    print(f'\n\nJVGAN parameters: Latent Dimension = {latent_dim}, num_features = {num_features}, lr={lr}', end="\n\n")


    # # Check if the models exist
    # generator_model_path = 'generator.pth'
    # discriminator_model_path = 'discriminator.pth'

    # # Initialize the models if they don't exist, otherwise load the saved models
    # if os.path.exists(generator_model_path) and os.path.exists(discriminator_model_path):
    #     # Load pre-trained models
    #     generator = Generator(latent_dim, num_features, num_classes).to(device)
    #     discriminator = Discriminator(num_features, num_classes).to(device)
        
    #     # Load the state dicts for both generator and discriminator
    #     generator.load_state_dict(torch.load(generator_model_path))
    #     discriminator.load_state_dict(torch.load(discriminator_model_path))
        
    #     print("Loaded pre-trained generator and discriminator models.")
    # else:
    #     # Initialize models if they don't exist
    #     generator = Generator(latent_dim, num_features, num_classes).to(device)
    #     discriminator = Discriminator(num_features, num_classes).to(device)
        
    #     print("Initialized new generator and discriminator models.")

    # Initialize models
    generator = Generator(latent_dim, num_features, num_classes).to(device)
    discriminator = Discriminator(num_features, num_classes).to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_data, real_labels) in enumerate(trainingDataloader):
            batch_size = real_data.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real data
            real_labels_onehot = nn.functional.one_hot(real_labels, num_classes).float().to(device)
            real_validity = discriminator(real_data, real_labels_onehot)
            d_real_loss = criterion(real_validity, torch.ones_like(real_validity).to(device))
            
            # Fake data
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_labels_onehot = nn.functional.one_hot(fake_labels, num_classes).float().to(device)
            fake_data = generator(z, fake_labels_onehot)
            fake_validity = discriminator(fake_data.detach(), fake_labels_onehot)
            d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity).to(device))
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_labels_onehot = nn.functional.one_hot(fake_labels, num_classes).float().to(device)
            fake_data = generator(z, fake_labels_onehot)
            fake_validity = discriminator(fake_data, fake_labels_onehot)
            g_loss = criterion(fake_validity, torch.ones_like(fake_validity).to(device))
            
            g_loss.backward()
            g_optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(trainingDataloader)}] "
                    f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


    # Testing
    # Convert to PyTorch tensors
    test_z = torch.FloatTensor(raw_data_testing.values[:, :-1]).to(device)
    test_labels = torch.LongTensor(raw_data_testing['target'].values).to(device)
    test_labels_onehot = nn.functional.one_hot(test_labels, num_classes).float().to(device)
    test_data = generator(test_z, test_labels_onehot)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

    # Evaluate the generated data for anomaly detection
    def evaluate_anomaly_detection(generator, real_data, labels, num_samples=1000):
        # Generate synthetic data
        z = torch.randn(num_samples, latent_dim).to(device)
        synthetic_labels = torch.randint(0, num_classes, (num_samples,)).to(device)
        synthetic_labels_onehot = nn.functional.one_hot(synthetic_labels, num_classes).float().to(device)
        synthetic_data = generator(z, synthetic_labels_onehot)
        
        # Combine real and synthetic data
        all_data = torch.cat([real_data, synthetic_data], dim=0)
        all_labels = torch.cat([labels, synthetic_labels], dim=0)
        
        # Use discriminator to classify real vs synthetic
        with torch.no_grad():
            predictions = discriminator(all_data, nn.functional.one_hot(all_labels, num_classes).float().to(device))
        
        # Convert predictions to binary (0 for synthetic, 1 for real)
        predictions = (predictions > 0.5).float()
        
        # Calculate anomaly detection metrics
        real_labels = torch.ones(real_data.size(0)).to(device)
        synthetic_labels = torch.zeros(synthetic_data.size(0)).to(device)
        true_labels = torch.cat([real_labels, synthetic_labels], dim=0)
        
        auc_roc = roc_auc_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels.cpu().numpy(), predictions.cpu().numpy(), average='binary')
        
        return auc_roc, precision, recall, f1


    # Evaluate anomaly detection performance
    auc_roc, precision, recall, f1 = evaluate_anomaly_detection(generator, features, labels)
    
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


    AUC_ROC.append(auc_roc)
    PRECISION.append(precision)
    RECALL.append(recall)
    F1_SCORE.append(f1)

print(f"AUC-ROC: {AUC_ROC}")
print(f"Precision: {PRECISION}")
print(f"Recall: {RECALL}")
print(f"F1-Score: {F1_SCORE}")

