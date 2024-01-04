import os
os.getcwd()

import numpy as np
from glob import glob

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

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8)) 
plt.bar(name_folder, num_images_per_folder)
plt.xticks(rotation='vertical')
plt.axhline(avg_no_images, color='black')
plt.xlabel('Dog breed folder')
plt.ylabel('Number of images in folder')
plt.title("Number of images per class- Black line corresponds to average number of images per dog breed")
#plt.show()

#######

import cv2                
import matplotlib.pyplot as plt                        
#%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


# TEST CASE No#1 FOR ONE PERSON IN IMAGE
# load color (BGR) image
#img = cv2.imread(human_files[6000])
img = cv2.imread("data/test_one_person.jpg")
#plt.imshow(img)
plt.show()
# convert BGR image to grayscale
#print("Grayscale Image")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray)
#plt.show()
# find faces in image
faces = face_cascade.detectMultiScale(gray) #gives bounding box coordinates

# print number of faces detected in the image
print('Number of faces detected in test picture No#1:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

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
plt.show()
# find faces in image
faces = face_cascade.detectMultiScale(gray)  # gives bounding box coordinates

# print number of faces detected in the image
print('Number of faces detected in test picture No#2:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# print(faces)

print(face_detector("./data/test_three_persons.jpg"))


from tqdm import tqdm

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

#######

import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


from PIL import Image
import torchvision.transforms as transforms

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

#print(VGG16_predict("data/klosar.jpg"))


#checking prediction on a sample image from dogs dataset
sample_output = VGG16_predict(dog_files[20])     #change VGG16_predict to return 'output' only
#print(sample_output)
#print('Max probability:', torch.max(sample_output[0]))

def dog_detector(img_path):
    ## TODO: Complete the function.
    prediction = VGG16_predict(img_path)
    return (prediction>=151 and prediction<=268)

#print(dog_detector("klosar.jpg"))


