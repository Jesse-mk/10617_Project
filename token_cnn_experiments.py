#!/usr/bin/env python
# coding: utf-8

# * https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# * https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# * https://pytorch.org/tutorials/beginner/saving_loading_models.html

# In[1]:

# In[2]:


import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import json


# In[10]:


IMAGE_SIZE = 96


# After unpickling, `train_set` and `test_set` will be lists, where each element is a dictionary that has keys `features` and `labels`. `features` will be a 1D numpy array of 1's and 0's, with size `box_size * box_size` where `box_size` is the size of the image. `label` will be a one-hot-encoded array.

# ### Generating Dataset 
# 

# In[3]:


class MathTokensDataset(Dataset):
    """
    Dataset containing math tokens extracted from the CROHME 2011, 2012, and 2013 datasets.
    """
    
    def __init__(self, pickle_file, image_size, transform=None):
        """
        Args:
            pickle_file (string): Path to dataset pickle file.
            transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        with open(pickle_file, 'rb') as f:
            self.df_data = pd.DataFrame(pickle.load(f))
        
        # Reshape features to 3D tensor.
        self.df_data['features'] = self.df_data['features'].apply(lambda vec: vec.reshape(1, image_size, image_size))
        
#         # Convert one-hot labels to numbers (PyTorch expects this).
#         self.df_data['label'] = self.df_data['label'].apply(lambda ohe_vec: np.argmax(ohe_vec))

        self.transform = transform
    
    def __len__(self):
        return len(self.df_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {
            'features': self.df_data.iloc[idx]['features'],
            'label': self.df_data.iloc[idx]['label']
        }
        
        if self.transform:
            sample = self.transform(sample)

        return sample


# In[4]:


class BaselineTokenCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineTokenCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 9 * 9, 600)
        self.fc2 = nn.Linear(600, 200)
        self.fc3 = nn.Linear(200, num_classes)
        
    def forward(self, x):
        x = x.float()
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[5]:


# Set device to GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# ##### Mods:
# 1. Changing Optimizers
# 2. Changing NN structure
# 3. ???
# 

# In[6]:


#### 1. Optimizers to try: ###
#we can add more but just wanted to see

optimizer_dict = {"adam": optim.Adam,
                  "sgd": optim.SGD,
                  "adamW": optim.AdamW}

optimizer_params_dict = {"adam": {"lr": 0.001,
                             "weight_decay": 0},
                    "sgd": {"lr": 0.001, 
                            "momentum": 0.9},
                    "adamW": {"lr": 0.001,
                    "weight_decay": 0.01 }}


# In[27]:


class Experiment():
    def __init__(self, experiment_name, optimizer_class, train_set, val_split, test_set, classes, batch_size, save_dir):
        #get runtime:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #get name for save files:
        self.experiment_name = experiment_name

        #make CNN
        self.net = BaselineTokenCNN(num_classes=len(classes))
        self.net.to(device)  # Send to GPU.

        #make loss
        self.criterion = nn.CrossEntropyLoss()

        #get optimizer and params:
        optimizer = optimizer_dict[optimizer_class]
        optimizer_params = optimizer_params_dict[optimizer_class]
        #add in the parameters:
        optimizer_params["params"] = self.net.parameters()
        # print(optimizer_params)

        #add in parameters to optimizer:
        self.optimizer = optimizer([optimizer_params])

        #keep track of train_history
        self.train_loss_history = []
        print("Model created with optimizer {}".format(optimizer_class))
        
        self.init_dataloaders(train_set, val_split, test_set, batch_size)
        
        print(f'{len(classes)} classes.')
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.save_dir = save_dir
        
        # Save the experiment settings.
        exp_dir = os.path.join(self.save_dir, self.experiment_name)
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        
        settings = {
            'optimizer': self.optimizer.state_dict(),
            'batch_size': batch_size,
            'val_split': val_split
        }
        
        settings_path = os.path.join(self.save_dir, self.experiment_name, 'settings.json' )
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        print(f'Initialized experiment \'{self.experiment_name}\'')
        
        
    def init_dataloaders(self, train_set, val_split, test_set, batch_size):
        if val_split is None or val_split == 0:
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            # Split the training set into train/validation.
            # Creating data indices for training and validation splits:
            num_train = len(train_set)
            indices = np.arange(num_train)
            split = int(np.floor(val_split * num_train))  # Index to split at.
            
            # Uncomment the line below if you want the train/val split to be different every time.
            # np.random.shuffle(indices)
            
            train_indices, val_indices = indices[split:], indices[:split]

            # Create PyTorch data samplers and loaders.
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            self.train_loader = torch.utils.data.DataLoader(train_set, 
                                                            batch_size=batch_size, 
                                                            sampler=train_sampler,
                                                            num_workers=4)
            self.val_loader = torch.utils.data.DataLoader(train_set,
                                                          batch_size=batch_size,
                                                          sampler=val_sampler,
                                                          num_workers=4)
            
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f'{len(train_indices)} training examples.')
        print(f'{len(val_indices)} validation examples.')
        print(f'{len(test_set)} test examples.')
        
    def train(self, max_epochs, patience):        
        best_val_loss = np.inf
        no_up = 0

        for epoch in tqdm(range(max_epochs), desc='Max Epochs'):
            for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training Batches', leave=False):
                # Get the inputs and send to GPU if available.
                features = data['features'].to(self.device)
                labels = data['label'].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            train_loss, train_acc = self.evaluate(self.train_loader, tqdm_desc='Eval. Train')
            val_loss, val_acc = self.evaluate(self.val_loader, tqdm_desc='Eval. Val')
            
            # Save statistics to history.
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            else:
                no_up += 1
                
                if no_up == patience:
                    self.save_checkpoint(epoch, val_loss)
                    print(f'Stopping after {epoch} epochs.')
                    print(f'Early stopping condition met, validation loss did not decrease for {patience} epochs.')
                    break
                
                
    def evaluate(self, dataloader, tqdm_desc=''):
        num_correct = 0
        num_total = 0
        
        total_loss = 0
        
        with torch.no_grad():
            for data in tqdm(dataloader, desc=tqdm_desc, leave=False):
                # Get the inputs and send to GPU if available.
                features = data['features'].to(self.device)
                labels = data['label'].to(self.device)

                # Get the predictions / loss.
                outputs = self.net(features)
                _, predicted = torch.max(outputs.data, dim=1)
                
                loss = self.criterion(outputs, labels)
                
                # Update correct/total counts.
                num_correct += (predicted == labels).sum().item()
                num_total += labels.size()[0]
                
                # Update total loss.
                total_loss += loss.item()
                
        acc = num_correct / num_total * 100
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, acc
                    

    def train_loss(self):
        # TODO: Is this correct? Should we really be averaging the train loss over all epochs?
        loss = np.mean(self.train_loss_history)
        print(f"Loss of the network on train set: {loss}")
        return loss

        
    def test_accuracy(self, classes, test_loader):
      
        self.num_classes = len(classes)

        self.total_counts = np.zeros(self.num_classes)
        self.correct_counts = np.zeros(self.num_classes)
        self.predictions = []
        # print(total_counts)
        # print(correct_counts)

        self.num_correct = 0
        self.num_total_examples = 0

        with torch.no_grad():
            for test_data in tqdm(test_loader):
                test_features = test_data['features'].to(self.device)
                labels = test_data['label'].to(self.device)
                
                outputs = self.net(test_features)
                
                _, predicted = torch.max(outputs.data, dim=1)
                self.predictions.append(predicted)
                
                for p, l in zip(labels, predicted):
                    self.total_counts[l] += 1
                    if p == l:
                        self.correct_counts[p] += 1
                
                self.num_total_examples += labels.size(0)
                self.num_correct += (predicted == labels).sum().item()
            self.test_accuracy = self.num_correct / self.num_total_examples * 100
        print(f'Accuracy of the network on test set: {self.test_accuracy}%')

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_dir = os.path.join(self.save_dir, self.experiment_name, 'checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        path = os.path.join(checkpoint_dir, f'epoch={epoch}_valLoss={np.round(val_loss, 4)}.pt')
        torch.save(self.net.state_dict(), path)
        
    def save_history(self):
        history_path = os.path.join(self.save_dir, self.experiment_name, 'history.csv' )
        pd.DataFrame(self.history).to_csv(history_path)
        
    def save_test_performance(self):
        test_loss, test_acc = self.evaluate(self.test_loader, tqdm_desc='Eval. Test')
        
        print(f'Test accuracy = {test_acc}%')
        
        test_perf_path = os.path.join(self.save_dir, self.experiment_name, 'test.json' )
        with open(test_perf_path, 'w') as f:
            json.dump({'test_loss': test_loss, 'test_acc': test_acc}, f)        


# In[28]:


def run_experiment(name, tokens_dataset_folder):
    prefix = os.path.join(os.getcwd(), 'data', 'tokens', tokens_dataset_folder)

    train_path = os.path.join(prefix, 'train.pickle')
    test_path = os.path.join(prefix, 'test.pickle')
    int_to_token_path = os.path.join(prefix, 'int_to_token.pickle')
    
    train_set = MathTokensDataset(train_path, IMAGE_SIZE)
    test_set = MathTokensDataset(test_path, IMAGE_SIZE)

    with open(int_to_token_path, 'rb') as f:
        int_to_token = pickle.load(f)
        
    classes = list(int_to_token.values())

    exp = Experiment(experiment_name=name, 
                     optimizer_class='adamW', 
                     train_set=train_set, 
                     val_split=0.2, 
                     test_set=test_set, 
                     classes=classes,
                     batch_size=4,
                     save_dir=os.path.join(os.getcwd(), 'experiments', 'token_cnn'))
    exp.train(max_epochs=100, patience=3)
    exp.save_history()
    exp.save_test_performance()


# In[29]:

# In[ ]:

if __name__ == '__main__':
    run_experiment(name='t=3,5,7', tokens_dataset_folder='b=96_train=2011,2013_test=2012_c=all_t=3,5,7')


# In[ ]:




