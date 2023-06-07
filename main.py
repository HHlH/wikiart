import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os.path


class ImageDataset(Dataset):
    def __init__(self, df, load_dir, transform):
        self.df = df
        self.load_dir = load_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.load_dir + "/Data/" + self.df.iloc[index,0]
        image = Image.open(image_path)
        tensor_image = self.transform(image)
        label = self.df.iloc[index,1]
        return tensor_image, label

# Model 1
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) # input, output, kernal size 
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*116*116, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        x = x.view(-1, 32*116*116)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x

# Model 2
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) # input, output, kernal size 
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32*114*114, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84,10)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        self.dp3 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)
        #print(x.shape)
        x = x.view(-1, 32*114*114)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp3(x)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x
  
num_workers = 12
num_epoch = 10
batch_size = 128
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dir = ""
df_train = pd.read_csv(load_dir+'/Labels/Genre/genre_train.csv')
df_val = pd.read_csv(load_dir+'/Labels/Genre/genre_val.csv')
transform = transforms.Compose([transforms.Resize((240, 240)),
                                transforms.ToTensor(),])
data_train = ImageDataset(df_train, load_dir, transform)
data_val = ImageDataset(df_val, load_dir, transform)
dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store accuracy values
train_acc_list = []
val_acc_list = []


for epoch in range(num_epoch):
    batch_accuracies = []
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader_train)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0) * 100
        train_acc_list.append(accuracy)
        batch_accuracies.append(accuracy)
    
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels in dataloader_val:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    val_accuracy = total_correct / total_samples * 100
    val_acc_list.append(val_accuracy)
    for count, accuracy in enumerate(batch_accuracies):
        if count % int(len(dataloader_train)/10) == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}], Batch [{count}/{len(dataloader_train)}], Train Accuracy: {accuracy:.2f}%")
    print(f"Epoch [{epoch+1}/{num_epoch}], Validation Accuracy: {val_accuracy:.2f}%")
    model.train()     
