import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
os.getcwd()


# load filenames for human and dog images
human_files = np.array(glob("./data/lfw/*/*"))
dog_files = np.array(glob("./data/dog_images/*/*/*"))

# print number of images in each dataset
#print('There are %d total human images.' % len(human_files))
#print('There are %d total dog images.' % len(dog_files))

#print('Total number of human face images:',len(glob("./data/lfw/*/*")))
#print('Total number of human face folders:', len(glob('./data/lfw/*')))
#print("Total numner of folders in 'dog_images:'", len(glob('./data/dog_images/*')))
#print("Folders in 'dog_images':",end=' ')
#print(*[x.split('/')[-1] for x in glob('./data/dog_images/*')], sep=',')
#print("Total folders(breed classes) in 'train, test, valid'",len(glob("./data/dog_images/train/*")))
#print('Total images in /dog_images/train :',len(glob("./data/dog_images/train/*/*")))
#print('Total images in /dog_images/test :',len(glob("./data/dog_images/test/*/*")))
#print('Total images in /dog_images/valid :',len(glob("./data/dog_images/valid/*/*")))

num_images_per_folder = [len(glob(x+'/*')) for x in glob("./data/dog_images/train/*")]
name_folder = [x.split('/')[-1] for x in glob("./data/dog_images/train/*")]

avg_no_images = sum(num_images_per_folder)/len(num_images_per_folder)
#print(avg_no_images)

plt.figure(figsize=(10,8))
plt.bar(name_folder, num_images_per_folder)
plt.xticks(rotation='vertical')
plt.axhline(avg_no_images, color='black')
plt.xlabel('Dog breed folder')
plt.ylabel('Number of images in folder')
plt.title("Number of images per class- Black line corresponds to average number of images per dog breed")



########################################################################################################################



# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


# TEST CASE No#1 FOR ONE PERSON IN IMAGE
# load color (BGR) image
#img = cv2.imread(human_files[6000])
img = cv2.imread("data/test_one_person.jpg")
#plt.imshow(img)
#plt.show()
# convert BGR image to grayscale
#print("Grayscale Image")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray)
#plt.show()
# find faces in image
faces = face_cascade.detectMultiScale(gray) #gives bounding box coordinates
print('Number of faces detected in test picture No#1:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
#plt.show()

#print(faces)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

print(face_detector("./data/test_one_person.jpg"))

# TEST CASE No#2 FOR THREE PERSONS IN IMAGE
# load color (BGR) image
# img = cv2.imread(human_files[6000])
img = cv2.imread("data/test_three_persons.jpg")
# plt.imshow(img)
#plt.show()
# convert BGR image to grayscale
#print("Grayscale Image")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
#plt.show()
# find faces in image
faces = face_cascade.detectMultiScale(gray)  # gives bounding box coordinates
print('Number of faces detected in test picture No#2:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
#plt.show()

# print(faces)

print(face_detector("./data/test_three_persons.jpg"))


# HUMAN AND DOG DETECTION
########################################################################################################################


human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.

#human_files_short_tq = tqdm(human_files[:100])
#dog_files_short_tq = tqdm(dog_files[:100])

#human_faces_in_humanData = list(map(face_detector, human_files_short_tq))
#human_faces_in_dogData = list(map(face_detector, dog_files_short_tq))

#print('Percentage of images in human_files that have detected human face:', (sum(human_faces_in_humanData)/len(human_faces_in_humanData))*100)
#print('Percentage of images in dog_files that have detected human face(incorrect detection):', (sum(human_faces_in_dogData)/len(human_faces_in_dogData))*100)


#######################################################################################################################



# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()



def load_transform_image(img_path):
    img = Image.open(img_path).convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  
    img_transform = transforms.Compose([
                        transforms.Resize(size=(224, 224)),    #VGG16 is trained on (244,244) images
                        transforms.ToTensor(),
                        normalize])
    img = img_transform(img)[:3,:,:].unsqueeze(0)
    return img
    
    
def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path

    image = load_transform_image(img_path)
    
    if use_cuda:
        image = image.cuda()
    output = VGG16(image)
    
    ## Return the *index* of the predicted class for that image
#     return np.argmax(output.detach().numpy()[0])
#     return output
# following return statement can also be used 
    return torch.max(output,1)[1].item()

#print(VGG16_predict("data/test_one_person.jpg"))


#checking prediction on a sample image from dogs dataset
sample_output = VGG16_predict(dog_files[20])     #change VGG16_predict to return 'output' only
#print(sample_output)
#print('Max probability:', torch.max(sample_output[0]))

def dog_detector(img_path):
    ## TODO: Complete the function.
    prediction = VGG16_predict(img_path)
    return (prediction>=151 and prediction<=268)

# TEST CASES FOR DOGS AND HUMANS TO DISTINGUISH IF A DOG EXISTS ON THE PICTURE
#print("The test No#1 is a dog on the image?",dog_detector("data/test_one_person.jpg"))
#print("The test No#2 is a dog on the image?",dog_detector("data/dog_images/test/065.Entlebucher_mountain_dog/Entlebucher_mountain_dog_04561.jpg"))


#########

from tqdm import tqdm

#human_files_short_tq = tqdm(human_files[:100])
#dog_files_short_tq = tqdm(dog_files[:100])

#dogs_in_humanData = list(map(dog_detector, human_files_short_tq))
#dogs_in_dogData = list(map(dog_detector, dog_files_short_tq))

#print('Percentage of dogs detected in human_files(incorrect detection):', (sum(dogs_in_humanData)/len(dogs_in_humanData))*100)
#print('Percentage of dogs detected in dog_files(correct detection):', (sum(dogs_in_dogData)/len(dogs_in_dogData))*100)

###

import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

data_dir = './data/dog_images/'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'valid/')
test_dir = os.path.join(data_dir, 'test/')

import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess_data = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize]),
                   'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                     transforms.ToTensor(), 
                                     normalize])
                  }

train_data = datasets.ImageFolder(train_dir, transform=preprocess_data['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=preprocess_data['valid'])
test_data = datasets.ImageFolder(test_dir, transform=preprocess_data['test'])

#print(train_data)
#print(valid_data)
#print(test_data)

batch_size = 20
num_workers = 0
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}

###

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        # Conv Layers
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # maxpool
        self.pool = nn.MaxPool2d(2, 2)
        # fc layers
        self.fc4 = nn.Linear(7*7*128, 2048)
        self.fc5 = nn.Linear(2048, 512)
        self.fc6 = nn.Linear(512, 133)    #number of classes = 133
        # dropout 
        self.dropout = nn.Dropout(0.25)    #dropout of 0.25
        # batchNorm layers
#         self.batch_norm_1 = nn.BatchNorm1d(2048)
        self.batch_norm = nn.BatchNorm1d(512)
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # flatten
        x = x.view(-1, 7*7*128)
#         x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.batch_norm(self.fc5(x)))
#         x = F.relu(self.batch_norm_2(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()

print(model_scratch)

###

import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr = 0.02)


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # watch training
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# train the model
#model_scratch = train(15, loaders_scratch, model_scratch, optimizer_scratch, 
#                      criterion_scratch, use_cuda, 'model_scratch.pt')

#print(model_scratch)

###

model_scratch.load_state_dict(torch.load('model_scratch.pt'))


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    
    preds = []
    targets = []
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        preds.append(pred)
        targets.append(target)
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    return preds, targets


#preds, gts = test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


#predictions = []
#for x in preds:
#    for y in np.array(x):
#        predictions.append(y[0])
#
#ground_truths = []
#for x in gts:
#    for y in np.array(x):
#        ground_truths.append(y)
#
#from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
#cm = confusion_matrix(ground_truths, predictions)
#precision = np.mean(precision_recall_fscore_support(ground_truths, predictions)[0])
#recall = np.mean(precision_recall_fscore_support(ground_truths, predictions)[1])
#print("Precision for breed classifier over test data:", precision)
#print("Recall for breed classifier over test data:", recall)
# print(metrics.classification_report(ground_truths, preds))

###

## TODO: Specify data loaders
loaders_transfer = loaders_scratch.copy()

import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
model_transfer = models.resnet101(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False
    
model_transfer.fc = nn.Linear(2048, 133, bias=True)    #replacing last fc with custom fully-connected layer which should output 133 sized vector

fc_parameters = model_transfer.fc.parameters()         #extracting fc parameters
for param in fc_parameters:
    param.requires_grad = True

if use_cuda:
    model_transfer = model_transfer.cuda()

#print(model_transfer)

###

import torch.optim as optim

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.001)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # watch training
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model

#model_transfer = train(20, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

#print(model_transfer)

model_transfer.load_state_dict(torch.load('model_transfer.pt'))

###

def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    
    preds = []
    targets = []
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        preds.append(pred)
        targets.append(target)
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    return preds,targets

#preds, gts = test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
#
#
#predictions = []
#for x in preds:
#    for y in np.array(x):
#        predictions.append(y[0])
#
#ground_truths = []
#for x in gts:
#    for y in np.array(x):
#        ground_truths.append(y)
#
#
#from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
#cm = confusion_matrix(ground_truths, predictions)
#precision = np.mean(precision_recall_fscore_support(ground_truths, predictions)[0])
#recall = np.mean(precision_recall_fscore_support(ground_truths, predictions)[1])
#print("Precision for breed classifier over test data:", precision)
#print("Recall for breed classifier over test data:", recall)
#
#
#
#import seaborn as sn
#import pandas as pd
#df_cm = pd.DataFrame(cm, index = range(133),
#                  columns = range(133))
#plt.figure(figsize = (20,15))
#sn.heatmap(df_cm, annot=True)
# not visible properly but confusion matrix for the data is stored in a dataframe df_cm


###


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]


### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]

loaders_transfer['train'].dataset.classes[:5]


def predict_breed_transfer(model, class_names, img_path):
    # load the image and return the predicted breed
    img = load_transform_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]


#testing our data on the separately given test images 
#for img_file in os.listdir('./data/dog_images/test/001.Affenpinscher'):
#    img_path = os.path.join('data/dog_images/test/001.Affenpinscher', img_file)
#    predition = predict_breed_transfer(model_transfer, class_names, img_path)
#    print("original image of: {0}, ************** predition breed: {1}".format(''.join([i for i in img_path.split('/')[-1].split('.')[0].replace('_',' ') if not i.isdigit()]), predition))


###

### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    img = Image.open(img_path)
    
    if dog_detector(img_path):
        print('Dog Detected!')
        plt.imshow(img)
        plt.show()
        print(f'This is a "{predict_breed_transfer(model_transfer, class_names, img_path)}" kind of dog!')
    elif face_detector(img_path):
        print('Human Detected!')
        plt.imshow(img)
        plt.show()
        print(f'Interesting, this human looks like "{predict_breed_transfer(model_transfer, class_names, img_path)}"!')
    else:
        print('Ooops! Nothing to detect!')
        plt.imshow(img)
        plt.show()

#for img_file in os.listdir('./images'):
#    img_path = os.path.join('./images', img_file)
#    run_app(img_path)
#    print('')


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack(glob('images/*')):
    run_app(file)
    print('---------------------------------------------------------------')


