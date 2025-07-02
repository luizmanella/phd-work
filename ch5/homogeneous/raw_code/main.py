import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm

import numpy as np
import json

from Agent import Agent

DOWNLOAD = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 2
BATCH_SIZE = 128

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, residual=None):
        x = self.bn(self.conv(x))
        if residual is not None:
            x += residual
        return F.relu(x)

class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.num_classes = 10

        self.conv1 = ConvBlock(self.in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 128) # residual connection to this
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 512)
        self.conv7 = ConvBlock(512, 512)
        self.conv8 = ConvBlock(512, 512) # residual connection to this

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, self.num_classes)
        
        # self.classifier = nn.Sequential(nn.MaxPool2d(4), 
        #                                 nn.Flatten(), 
        #                                 nn.Linear(512, self.num_classes))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.pool1(x)
        x = self.conv3(residual)
        x = self.conv4(x, residual)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.conv6(x)
        residual2 = self.pool3(x)
        x = self.conv7(residual2)
        x = self.conv8(x, residual2)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# BASE_MODEL = ResNet9()
# AGENT_MODEL_POINTER = ResNet9
BASE_MODEL = Net()
AGENT_MODEL_POINTER = Net
LOSS_FUNCTION = nn.CrossEntropyLoss()

print('Grabbing trainset...')
with open('new_trainset.json', 'r') as f:
    trainset = json.load(f)
print('Finished grabbing Trainset')
new_trainset = []
for s in trainset:
    new_trainset.append((torch.tensor(s[0]), s[1]))

print('Grabbing testset...')
with open('new_testset.json', 'r') as f:
    testset = json.load(f)
print('Finished grabbing testset')
new_testset = []
# for s in trainset: what previously was
for s in testset:
    new_testset.append((torch.tensor(s[0]), s[1]))

trainset = new_trainset
testset = new_testset

training_data = [[] for _ in range(10)]
testing_data = [[] for _ in range(10)]
for sample in tqdm(trainset):
    training_data[sample[1]].append(sample)
for sample in tqdm(testset):
    testing_data[sample[1]].append(sample)

# Agents
list_of_agents = []
number_of_agents = 100
num_tgt_agents = 100
train_data_per_agent = int((len(trainset)/number_of_agents)/10)
test_data_per_agent = int((len(testset)/number_of_agents)/10)
for a in range(number_of_agents):
    agent_model = AGENT_MODEL_POINTER()
    agent_model.load_state_dict(BASE_MODEL.state_dict())
    agent_optimizer = optim.Adam(agent_model.parameters(), lr=1e-3)
    # Create a new agent
    list_of_agents.append(Agent(id=a, device=DEVICE, number_of_epochs=EPOCHS))
    list_of_agents[-1].set_model(agent_model)
    list_of_agents[-1].set_loss_function(LOSS_FUNCTION)
    list_of_agents[-1].set_optimizer(agent_optimizer)

    # Create the dataset for that agent
    _train_dataset = []
    _test_dataset = []

    # loop through each class type
    if a < number_of_agents-1:
        for i in range(10):
            # pull random samples from class i
            rand_idx = np.random.randint(len(training_data[0])-10, size=train_data_per_agent)
            rand_idx.sort()
            for idx in rand_idx[::-1]:
                idx_sample = training_data[i].pop(idx)
                _train_dataset.append(idx_sample)

            rand_idx = np.random.randint(len(testing_data[0])-10, size=test_data_per_agent)
            rand_idx.sort()
            for idx in rand_idx[::-1]:
                idx_sample = testing_data[i].pop(idx)
                _test_dataset.append(idx_sample)
    else:
        for i in range(10):
            _train_dataset.extend(training_data[i])
            _test_dataset.extend((testing_data[i]))
    agent_train_dataloader = torch.utils.data.DataLoader(_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    agent_test_dataloader = torch.utils.data.DataLoader(_test_dataset, batch_size=BATCH_SIZE)
    list_of_agents[-1].set_training_data(agent_train_dataloader)
    list_of_agents[-1].set_testing_data(agent_test_dataloader)


global_model_test_acc = []
training_loops = 1000
for _ in range(training_loops):
    # Train model
    target_agents_idx = np.random.randint(len(list_of_agents), size=(num_tgt_agents,)).tolist()
    target_agents = [list_of_agents[i] for i in target_agents_idx]
    for agent in target_agents:
        agent.train_model()
    print("Finished training Models")
    # Average models
    base_model_state_dict = BASE_MODEL.state_dict()
    for key in base_model_state_dict:
        if key.endswith('num_batches_tracked'):
            continue
        first_agent_state_dict = list_of_agents[0].grab_model_parameters()
        param_avg = first_agent_state_dict[key]
        for agent in list_of_agents[1:]:
            param_avg += agent.grab_model_parameters()[key]
        param_avg /= number_of_agents
        base_model_state_dict[key] = param_avg
    BASE_MODEL.load_state_dict(base_model_state_dict)
    for agent in list_of_agents:
        agents_model = agent.get_model()
        agents_model.load_state_dict(BASE_MODEL.state_dict())
        agent.set_model(agents_model)

    # AFTER PERFORMING MODEL AGGREGATION - TEST THE MODEL AND STORE
    final_avg_acc = 0
    for agent in list_of_agents:
        final_avg_acc += agent.test_model(store_results=False)
    final_avg_acc /= len(list_of_agents)
    global_model_test_acc.append(final_avg_acc)
    print(f"Training Loop: {_} <--> Final Average Accurary: {final_avg_acc}")

    print('Storing current results...')
    agents_training_lists = {}
    agents_testing_lists = {}
    for agent in list_of_agents:
        agents_training_lists[f"Agent {agent.grab_id()}"] = agent.get_training_accuracy()
        agents_testing_lists[f"Agent {agent.grab_id()}"] = agent.get_testing_accuracy()

    target_json_to_store = {
        "local_agents_training": agents_training_lists,
        "local_agents_testing": agents_testing_lists,
        "global_testing": global_model_test_acc
    }
    with open(f'fl_simulation_random_selection_results_{number_of_agents}_agents_{num_tgt_agents}_target_agents.json', 'w') as f:
        json.dump(target_json_to_store, f)
    print('Finished storing results!')