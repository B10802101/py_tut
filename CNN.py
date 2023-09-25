import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import time
num_epochs = 100
batch_size = 100
learning_rate = 0.001
num_folds = 5

class CNN(nn.Module):                           #Inherit from pytorch module
    def __init__(self):                         #Constructor, to init the param of the model
        super(CNN, self).__init__()             #The constructor is inherited from nn.module
        self.conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=3)    #26*26*3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                       #13*13*3
        self.conv2 = nn.Conv2d(in_channels=3, kernel_size=3, out_channels=9)    #11*11*9
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(11*11*9, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self,x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        # x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.conv2(x)
        x = torch.flatten(x,1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)
# train_size = int(0.8*len(train_dataset))
# valid_size = len(train_dataset) - train_size
# train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_loss = []
best_accuracy = 0
for fold, (train_index, valid_index) in enumerate(kf.split(train_dataset.data, train_dataset.targets)):
    print(f'Fold {fold + 1}/{num_folds}')
    train_subset = torch.utils.data.Subset(train_dataset, train_index) 
    valid_subset = torch.utils.data.Subset(train_dataset, valid_index)
    train_loader = DataLoader(dataset = train_subset, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_subset, batch_size = batch_size, shuffle = True)
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # init optimizer
            optimizer.zero_grad()
            
            # forward -> backward -> update
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()

            training_loss.append(loss.item())
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            valid_accuracy = 100.0 * correct / total
            print('Validation Accuracy: {}'.format(valid_accuracy))
            if valid_accuracy > best_accuracy:
                print('saving better model...')
                best_accuracy = valid_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
        print("training time of this epoch is: ", time.time() - start_time)

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        #for batch size = 1
        # label = labels.item()
        # pred = predicted.item()
        # if (label == pred):
        #     n_class_correct[label] += 1
        # n_class_samples[label] += 1

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i}: {acc} %')
plt.plot(training_loss, label = 'training loss')
plt.xlabel('training step')
plt.ylabel('training loss')
plt.legend()
plt.show()