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
from sklearn.metrics import confusion_matrix
import numpy as np
num_epochs = 100
batch_size = 1000
learning_rate = 0.001
num_folds = 5
earlystop_patience = 70
earlystop_record = []
data_transform_type = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
true_labels = []
predict_labels = []
class CNN(nn.Module):                           #Inherit from pytorch module
    def __init__(self):                         #Constructor, to init the param of the model
        super(CNN, self).__init__()             #The constructor is inherited from nn.module
        self.conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=3)    #26*26*3
        self.conv2 = nn.Conv2d(in_channels=3, kernel_size=3, out_channels=9)    #22*22*9
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                       #11*11*9
        self.conv3 = nn.Conv2d(in_channels=9, kernel_size=3, out_channels=27)    #20*20*27
        self.fc1 = nn.Linear(10*10*27, 400)
        self.fc2 = nn.Linear(400, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x,1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x
model = CNN()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=data_transform_type, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=data_transform_type, download=False)
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
training_accuracy = []
validation_accuracy = []
validation_loss = []
best_accuracy = 0
for fold, (train_index, valid_index) in enumerate(kf.split(train_dataset.data, train_dataset.targets)):
    print(f'Fold {fold + 1}/{num_folds}')
    train_subset = torch.utils.data.Subset(train_dataset, train_index) 
    valid_subset = torch.utils.data.Subset(train_dataset, valid_index)
    train_loader = DataLoader(dataset = train_subset, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_subset, batch_size = batch_size, shuffle = True)
    n_total_steps = len(train_loader)
    model_no_improve = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # init optimizer
            optimizer.zero_grad()
            
            # forward -> backward -> update
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()

            optimizer.step()

            total_train += labels.size(0)
            _, train_predicted = torch.max(outputs.data, 1)
            correct_train += (train_predicted == labels).sum().item() 

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        training_loss.append(loss.item())
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        print('         Training Accuracy: {}'.format(train_accuracy))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            valid_loss = 0.0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
            valid_accuracy = 100.0 * correct / total
            validation_accuracy.append(valid_accuracy)
            validation_loss.append(loss.item())
            print('         Validation Loss: {}'.format(loss.item()))
            print('         Validation Accuracy: {}'.format(valid_accuracy))
            if valid_accuracy > best_accuracy:
                print('saving better model...')
                best_accuracy = valid_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                model_no_improve += 1
                if model_no_improve >= earlystop_patience:
                    print('Model not improved, early stop after {} epochs'.format(epoch + 1))
                    earlystop_record.append(model_no_improve)
                    break
        print("training time of this epoch is: ", time.time() - start_time)

print('Finished Training')
model.load_state_dict(torch.load('best_model.pth'))

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    mismatch_images = []
    mismatch_indexes = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())                     #move data back on the CPU to operate   
        predict_labels.extend(predicted.cpu().numpy())               #move data back on the CPU to operate 

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            else: 
                mismatch_images.append(images[i].cpu())
                mismatch_indexes.append([label, pred])
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i}: {acc} %')


fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))
ax1.plot(range(len(training_loss)), training_loss, 'b', label = 'training loss')
ax1.plot(range(len(validation_loss)), validation_loss, 'g', label = 'validation loss')
ax1.set_xlabel('Number of steps')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(range(len(validation_accuracy)), validation_accuracy, 'g', label = 'validation accuracy')
ax2.plot(range(len(training_accuracy)), training_accuracy, 'b', label = 'training accuracy')
ax2.legend()

conf_matrix = confusion_matrix(true_labels, predict_labels)         #confusion matrix
plt.figure(figsize = (8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = [str(i) for i in range(10)]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation = 45)
plt.xticks(tick_marks, classes)
fmt = 'd'
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.figure(figsize = (10, 10))

image = np.array([mismatch_images[i].squeeze().numpy() for i in range(len(mismatch_images))])
fig, ax = plt.subplots(11, 11, figsize=(10, 10))
for i in range(len(mismatch_images)):
    row, col = divmod(i, 11)
    ax[row, col].imshow(image[i], cmap='gray')
    true_label, pred_label = mismatch_indexes[i]
    ax[row, col].set_title(f'T:{true_label}, F:{pred_label}', fontsize=5)
    ax[row, col].axis('off')
for i in range(len(mismatch_images), 121):
    row, col = divmod(i, 11)
    ax[row, col].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
