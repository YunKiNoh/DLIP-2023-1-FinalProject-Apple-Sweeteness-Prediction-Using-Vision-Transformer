# Apple Sweeteness Prediction Using Vision Transformer

 

**date**: 2023/06/19

**Author**: Yun Ki Noh/ EunChan Kim

**Github**: https://github.com/GracenPraise/DLIP2023

**Demo Video**: https://youtu.be/GrVAf8BUc9s

## 1. Introduction

In this project, we want to help cutomer to select apples when many apples are mixed without information that which apple is more sweet. Customer would want to spend money effectively, so they will spend more time to select apple. In addition to this, although there are several ways to check apple's sugar content(for example, using juice or infrared light), each way has limitation about damage to apple or high cost. To predict apple's sugar content by brix witout damge to apple and expencise cost, we determine using deep learing model. 

### Goal

By trainig deep learning model to predict sugar content of apple(Brix), predicting apple's sugar content in real time and print out the result is available. Since there have been no preceding studies on predicting apple sweetness, the train loss was set to be within 2 MSE (Mean Squared Error) losses, and the error between the actual sweetness value and the predicted value was arbitrarily set to be within plus or minus 5%.

### Hardware

* NVIDIA GeForce RTX 3050

* S604HD VER 1.0

* Server Computer(DSTECH)

  | Device | Specificaition                      |
  | ------ | ----------------------------------- |
  | GPU    | 1xAMD EPYC 7742 2.25GHz Upto 3.4GHz |
  | CPU    | 4EA x NVIDIA A100 80GB (320GB VRAM) |

  > Table 1. Server Computer

#### Hardware setting

<img width="435" alt="angle_picture" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/27504f9d-6e9a-4866-9981-8bb263b2b2ee">

<img width="479" alt="angle" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/855b377f-54f2-4cf4-a6d8-e0567ba2c8a8">

> **Figure 1. Camera's Angle & Distance**

* Camera angle: 50**°**

* Camera height: 25.0cm

  ​


### Software 

* Window 11

- OpenCV 3.9.13

- NVIDIA Driver Version 526.56

- CUDA 11.7

- PyTorch 1.12.1

- YOLOv8s-seg

  ​



## 2. Dataset

<img width="359" alt="dataset" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/1d6f28ab-d4e0-4bcb-9c91-57648fc9b617">

> **Figure 2. Deep Learning Dataset about apple from Jeonbuk**

We used Jeonbuk's apple dataset to train deep learning model. Because Huji apple is most consumed in south korea, we chose 17,000 images and labeling dataset of Huji apple.



### Format of dataset

Dataset from Jeonbuk is provided labeling data as json file. 

|      Sugar Content       |         Segmentation         |
| :----------------------: | :--------------------------: |
| <img width="195" alt="labeling" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/02cca91c-0e7d-4893-8477-231fec72f31c"> | <img width="468" alt="segmentation" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/1f42ea61-5bdf-4ae9-bc59-e426ef442e77">> |

> **Table 2. Labeling Data**

In the labeling dataset, we used 'segmentation coordination' to extract apple's pixels and 'sugar_content's information' to predict sugar content in Python code. 

**Dataset link**: [AI-Hub ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=490)



## 3. Tutorial Procedure

### Setup & Installation

To train deep learning model in Python, you have to setup several things. 

#### 1. Anaconda Virtual Environment

To deal with Python programming in private circumstance, you have to install several necessary programs in your private virtual environment. To create virtual environment, enter this code in anaconda.

``` 
conda create -n py39 python=3.9.13
```

If you want to activate, enter this code in Anaconda

``` 
conda activate py39
```

#### 2. Install Libs 

Install Numpy, OpenCV, Matplot, Jupyter in your virtual environment

```
conda activate py39
conda install -c anaconda seaborn jupyter
pip install opencv-python
pip install torch torchvision
pip install pillow
pip install numpy
pip install tqdm
pip install opencv-python
pip install matplotlib
pip install timm
pip install ultralytics
```

#### 3. Install Visual Studio Code

Install VS Code by checking this site. 

**link**: [VS Code Downloads](https://code.visualstudio.com/download)

#### 4-1. Server Computer's CUDA, cuDNN(Option) 

Create private virtual environment and install CUDA and PyTorch 

* CUDA 11.3.1
* PyTorch 1.10

#### 4-2. Install GPU Driver, CUDA, cuDNN

Because each driver has proper versions for compatbility, you must select the appropriate version of GPU Driver, CUDA, cuDNN.

**1. Check PyTorch & CUDA Version**

<img width="635" alt="pytorch" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/3595a722-11ec-420f-b1a8-d94f590116bb">

> **Figure 3. Pytorch & CUDA Version**

To use PyTorch, we have to use 11.7 or 11.8 version of CUDA.

**2. Check Proper GPU Version for CUDA 11.7** 

<img width="733" alt="toolkit" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/bdc69e89-e0d0-40db-bb62-7f93d227200b">

> **Figure 4. CUDA Toolkit**

**Install GPU Driver**

<img width="387" alt="gpudriver" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/133fffb1-938d-41b8-9d89-66fa70ddd937">

> **Figure 5. GPU Driver**

By checking your computer's information, install GPU Driver. 

**link**:[GPU Driver](https://www.nvidia.co.kr/Download/index.aspx?lang=kr)

**3. Check Proper cuDNN Version for CUDA 11.7** 

<img width="1024" alt="cudnn" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/6a12ba30-1f3d-4380-ac23-cc2d8e04a1e3">

> **Figure 6. cuDNN version**

**link**:[cuDNN version](https://developer.nvidia.com/rdp/cudnn-archive)

**4. Install CUDA**

Install CUDA 11.7

**link**:[CUDA Toolkit 11.7 Downloads](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

**5. Install PyTorch**

Install PyTorch with CUDA 11.7 version

**link**:[PyTorch Downloads](https://pytorch.org/)

**6. Install cuDNN**

**link**:[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

## 4. Train & Test Process

### Train  Process

#### 1. Apple Segentation

When measuring the sweetness of an apple, it is necessary to predict the sweetness only within the apple region, rather than the entire image. Therefore, segmentation is essential. After trying various models, we found that the pretrained model provided by Yolov8s-seg performed the best in apple segmentation, so we utilized it. By using the pretrained model, we were able to invest more time in training.

<img width="486" alt="seg" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/5f51f480-090c-4774-ad52-19a6be9272cf">

> **Figure 7. Result of apple segmentation with yolov8-seg**

#### 2. Brix Prediction

To predict the sweetness of an apple, the model needs to be trained solely on apple images and corresponding sweetness labels, solving a regression problem for sweetness prediction. Therefore, we determined that using a sophisticated and highly accurate model is appropriate. We experimented with various models such as resNet50, DenseNet, VGG16, Inception v3, among others, but encountered significant loss compared to our expectations.

![models](https://github.com/GracenPraise/DLIP2023/assets/91367451/230886fc-1a29-49c1-902d-dfedb5d6f91f)

> **Figure 8. MES loss comparison of multiple deep learning model**

Therefore, we decided to utilize the Vision Transformer model, which has been widely used in deep learning for image processing recently. Specifically, we employed the vit_base_patch16_224 model, which accepts inputs of size 224x224 and utilizes a 16-patch approach.

![vit](https://github.com/GracenPraise/DLIP2023/assets/91367451/a96e86d7-e83c-4e67-9865-0d195376a184)

> **Figure 8. Structure of Vision Transformers (VIT). Referenced by 'Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.'**

In the provided apple dataset, there was an issue where the segmentation coordinates did not align correctly due to the application of rotation in certain photos. To address this problem, we performed preprocessing by extracting the EXIF information from the photos and adjusting them accordingly if rotation was applied, ensuring proper alignment

```python
def open_and_rotate_image(image_path):
    # if image is rotated correct the image rotation
    image = Image. open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image. rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image. rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image. rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If there's no EXIF ??data or the orientation data isn't set, just return the original image
        pass
    return image
```



Approximately 170,000 images were used for training, while validation utilized around 17,000 images. Due to the large number of training images, we increased the epoch size. However, this resulted in significantly longer training times. After observing no significant decrease in loss beyond 5 epochs, we set the number of epochs to 10. For loss measurement, we employed the Mean Squared Error (MSE) loss, commonly used in regression problems. The optimizer used was Adam.

```python
num_models = 3
    models_list = []
    for _ in range(num_models):
        # Train using vit_base_patch16_224 model and freeze parameters
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        # Update the model's head to output a single value
        model.head = nn.Linear(model.head.in_features, 1)
        
        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Add the model to the models list
        models_list.append(model)
    
    # Use the MSE loss
    criterion = nn.MSELoss()
    
    # Initialize best validation loss for each model
    best_loss = [float('inf')] * num_models  
    
    for i, model in enumerate(models_list):
        print(f"Training model {i+1}")
        
        # Initialize Adam optimizer 
        optimizer = optim.Adam(model.head.parameters())
    
        num_epochs = 10
        
        for i, model in enumerate(models_list):
            print(f"Training model {i+1}")
    
            # Create lists to store loss per epoch for training and validation
            train_losses = []
            val_losses = []
    
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                pbar = tqdm(train_dataloader)

                for inputs, labels in pbar:
                    # Move inputs and labels to the GPU if available
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    # Reshape the labels
                    labels = labels.view(-1, 1) 
                    # Zero out the gradients
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(inputs)
                    # Calculate the loss
                    loss = criterion(outputs, labels)
                    # Backward pass
                    loss.backward()
                    optimizer.step()
  
                    running_loss += loss.item() * inputs.size(0)
                    pbar.set_description(f"Train Epoch: {epoch+1}, Loss: {loss:.4f}")
                
                # Calculate and print epoch loss for training
                epoch_loss = running_loss / len(train_dataset)
                train_losses.append(epoch_loss)
                print(f'Train Loss: {epoch_loss:.4f}')
```

During the project, it was observed that the set target train loss of less than 2 was met, and it was confirmed that the training proceeded smoothly without any signs of overfitting.

![loss](https://github.com/GracenPraise/DLIP2023/assets/91367451/79ed37bb-f551-4ecc-baa8-2e9b5494548a)

> **Graph1. Train and Validation Loss Graph**



### Test Process

**Model Download: **[link](https://drive.google.com/drive/folders/12fvvtiPWf_keCtu5OOUQ4VUQA91ZJuhU?usp=drive_link)

The code was written to utilize the segmentation model and regression model developed in the above process, enabling real-time prediction of apple sweetness. The simple flow is as follows.

![flow](https://github.com/GracenPraise/DLIP2023/assets/91367451/331d7430-a64a-4ffe-b2d8-a3e19b6e990c)

> **Chart 1. Flow Chart**

First, the frame image is resized to a size of 224x224 to make it compatible with the prediction model. 

```python
# Define transformations for the input
input_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
```

Then, the yolov8-seg model is loaded to perform apple detection. If no apple is detected, the message 'Put the Apple' is displayed. 

```python
# Load segmentaion model
weight_path = 'yolov8s-seg.pt'

if ret == True: # Run YOLOv8 inference on the frame
        # Change the frame size
        results = model(frame, imgsz=224)
        result = results[0]
        len_result = len(result)

        cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

        if len_result == 0:
            # when no object in frame
            cv2.rectangle(frame, (0,0,w,h), (153,255,255), 7)
            cv2.putText(frame, f'Put the Apple', (230, 200), user_font, 0.7, (153,255,255), 4)
            cv2.putText(frame, f'Put the Apple', (230, 200), user_font, 0.7, (255,255,255), 2)
```

Once an object is detected and the Enter key is pressed, the segmentation process is initiated. As a result, a mask operation is performed to isolate the data within the apple region. Only the data within the segmented apple area is then utilized for further processing and analysis.

```python
# When stop predict Brix
if signal == 0:
    predicted_sugar_contents = []
    cv2.rectangle(frame, (0,0,w,h), (255,102,102), 7)
    cv2.putText(frame, f'Press Enter for Check the Brix', (130, 200), user_font, 0.7, (255,102,102), 4)
    cv2.putText(frame, f'Press Enter for Check the Brix', (130, 200), user_font, 0.7, (255,255,255), 2)

# When predict Brix
elif signal == 1:
    iteration = 0
    cv2.rectangle(frame, (0,0,w,h), (102,102,255), 7)
    cv2.putText(frame, f'Press r key for end check', (210, 400), user_font, 0.5, (102,102,255), 3)
    cv2.putText(frame, f'Press r key for end check', (210, 400), user_font, 0.5, (255,255,255), 2)

    # if Apple confidence over 0.7
    if cls == 47 and conf > 0.7:
        iteration += 1

        segment = detection.masks.xy[0].astype(np.int32)
        print("detection.masks",segment)

        color = colors[cls]
        if segment.shape[0] > 0:

            # Use only apple regions for prediction
            mask_base = np.zeros(frame.shape, dtype = np.uint8)
            cv2.fillPoly(mask_base, [segment], (255,255,255))
            img_gray = cv2.cvtColor(mask_base, cv2.COLOR_BGR2GRAY)
            mask2 = cv2.bitwise_and(frame, frame, mask = img_gray)

            # Get bounding box coordinates
            min_x, min_y = np.min(segment, axis=0)
            max_x, max_y = np.max(segment, axis=0)

            # Draw the bounding rectangle
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (102,102,255), 3)  # you can change the color and thickness
```

The received data within the segmented area is used for the prediction of apple sweetness. The predicted sweetness value is then displayed.

```python
def predict_sugar_content(_frame):
    # Convert the numpy array frame to a PIL image
    img = Image.fromarray(_frame).convert('RGB')

    # Apply transformations
    img = input_transforms(img)
    img = img.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model_trans(img)
        predicted = output.item()
        return predicted

model_path = "trans1.pth"  
model_trans = timm.create_model('vit_base_patch16_224', pretrained=False)  
model_trans.head = nn.Linear(model_trans.head.in_features, 1) 
model_trans.load_state_dict(torch.load(model_path)) 
model_trans.eval() 

# In your main loop, after predicting the sugar content, append it to the deque
predicted_sugar_content = round(predict_sugar_content(mask2), 2)  
predicted_sugar_contents.append(predicted_sugar_content)

# Calculate the average predicted sugar content from the values in the deque
avg_predicted_sugar_content = sum(predicted_sugar_contents) / len(predicted_sugar_contents)

cv2.putText(frame, f"Brix: {avg_predicted_sugar_content:.2f}", (min_x+70, min_y+100), user_font, 0.5, (102,102,255), 2)
cv2.putText(frame, f"Brix: {avg_predicted_sugar_content:.2f}", (min_x+70, min_y+100), user_font, 0.5, (255,255,255), 1)
```

Finally, the results are displayed on the screen. Pressing the 'r' key takes you back to the previous step of segmentation.



Due to the heavy computational load of the model used for real-time apple sweetness prediction, significant frame drops occur. To mitigate this issue, the prediction is performed only when the apple is placed in position and the Enter key is pressed. To prevent the fluctuation of values with each frame change, the average sweetness value from multiple frames is used. The accumulated sweetness values are continuously added to calculate the average. If the apple is changed, the 'r' key should be pressed to reset the accumulation before proceeding with prediction by pressing the Enter key again.

| ![ready](https://github.com/GracenPraise/DLIP2023/assets/91367451/ba8f6e11-ec82-4565-bc6d-8b688a37ba02) | ![press_enter](https://github.com/GracenPraise/DLIP2023/assets/91367451/2fb3adb9-62b7-492c-8600-97b9e51f096f) | ![result](https://github.com/GracenPraise/DLIP2023/assets/91367451/0f0d6a77-ac59-4752-9504-66418f435a9b) |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
|  **Figure 9. When No Object in Frame**   |    **Figure 10. When Apple Deteted**     |      **Figure 11. Brix Prediction**      |



## 5. Results and Analysis

### Results

![result_brix](https://github.com/GracenPraise/DLIP2023/assets/91367451/7102e510-d952-4672-9265-2948cf7be08f)

> **Figure 12. Result**

Our project's final result is predicted brix value of apple. Before utilizing this brix value, checking this value is correct is essential.

To check reliability of our trained model, we chose 10 apples and check the error between real value of brix and predicted value of our model. And the result is like this.

|                  |   1   |   2   |   3   |   4   |   5   |   6   |   7   |  8   |   9   |  10   |
| :--------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: | :---: | :---: |
| Real[brix]       | 12.5  | 13.0  | 13.7  | 11.0  | 12.0  | 10.0  | 14.2  | 11.0 | 13.0  | 12.0  |
| Prediction[birx] | 12.44 | 12.76 | 13.15 | 11.71 | 12.16 | 11.53 | 14.12 | 11.7 | 12.88 | 13.63 |
| Error[%]         | 0.48  | 1.85  | 4.01  | 6.45  | 1.33  | 15.3  | 0.56  | 6.36 | 0.92  | 13.58 |

| Total[%] |
| :------: |
|   5.08   |

There were significant errors in a few cases,  so we couldn't achieve the goal of an error within 5%. 

> **Table 3. Evaluation Error**

|             | Train Error[MSE] | Validation Error[MSE] | Evaluation Error[%] |
| :---------: | :--------------: | :-------------------: | :-----------------: |
|    Goal     |     Under 2      |    No Overfitting     |      Under 5%       |
|   Result    |       1.89       |         3.02          |        5.08         |
| Achievement |        O         |           O           |          X          |

> **Table 4. Result**

In the process of training model, we can check train loss under 2 MSE loss and this model is not overfitted by checking validation error. However, we couldn't achieve evaluation goal. 



### Analysis

#### Light condition

Upon examining the data used for training, it was observed that more than 70% of the images were captured in an orchard. This indicates that the photos were taken under sufficient lighting conditions, implying that evaluating the model trained indoors may introduce errors. To mitigate this, it is expected that increasing the quantity of training data captured under indoor conditions would lead to improved performance.

<img width="576" alt="oneside" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/1fd6c890-8621-414b-b9b2-944864f42f6a">

> **Figure 13. Most of Images Taken in Outside**



#### Rotation  

Furthermore, it was observed that when the camera and the apple underwent rotation, the sweetness value varied even for the same apple. This could be attributed to the fact that the apple images used for training represented only one side of the apple, potentially introducing errors. It is believed that training with multiple angle images for a given apple-sweetness dataset could help reduce such errors.

|![normal](https://github.com/GracenPraise/DLIP2023/assets/91367451/a46e0ca3-3834-4d11-b021-52bdb1668017) | ![rotation](https://github.com/GracenPraise/DLIP2023/assets/91367451/6a59645b-7784-4cbf-ad07-077b6fbf0f79) |
| :--------------------------------------: | :--------------------------------------: |
| **Figure 14. Before Rotation (12.26 Brix)** | **Figure 15. After Rotation (10.35 Brix)** |



## 6. Reference

- 강다영 외 5명. (2021). CNN 을 활용한 수박 당도 예측. ACK 2021.

- 채이한,한지훈. (2021). CNN 기반 이미지 분석을 통한 과일 당도 예측. 세종과학예술영재학교

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.

- Al-Sammarraie, M.A.J., Gierz, Ł.,  Przybył, K.,  Koszela, K., Szychta, M.,  Brzykcy, J.,  Baranowska,H.M. (2022). Predicting Fruit’s SweetnessUsing Artificial Intelligence—Case Study: Orange. Appl.

- Sangsongfa, A., Am-Dee, N., Meesad P.(2020). Prediction of Pineapple Sweetness from Images Using Convolutional Neural Network. EAI

  ​


## 7. Appendix

```python
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import timm
from collections import deque

# Initialize
predicted_sugar_content = 0
signal = 0

# Load segmentaion model
weight_path = 'yolov8s-seg.pt'

model = YOLO(weight_path)

# Cam setting
my_cam_index = 0
cap = cv2.VideoCapture(my_cam_index, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FPS, 30.0) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 
print(cap.get(cv2.CAP_PROP_EXPOSURE))

# ------------------------------------------------------------------- #

width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps     = int(cap.get(cv2.CAP_PROP_FPS))
print('get cam fps: ', fps)
user_font    = cv2.FONT_HERSHEY_COMPLEX

# Video write initialize
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = 'test.mp4'
frame_size = (width, height)
fps = cap.get(cv2.CAP_PROP_FPS) 
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

with open('counting_result.txt', 'w') as f:
    f.write('')

# Define transformations for the input
input_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

def predict_sugar_content(_frame):
    # Convert the numpy array frame to a PIL image
    img = Image.fromarray(_frame).convert('RGB')

    # Apply transformations
    img = input_transforms(img)
    img = img.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model_trans(img)
        predicted = output.item()
        return predicted


model_path = "trans1.pth"  
model_trans = timm.create_model('vit_base_patch16_224', pretrained=False)  
model_trans.head = nn.Linear(model_trans.head.in_features, 1) 
model_trans.load_state_dict(torch.load(model_path)) 
model_trans.eval()  


def key_command(_key):
             

        if _key == ord('s') :   cv2.waitKey()
        
        elif _key == ord('i'):
        # Get current exposure.
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            # Increase exposure by 1.
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure + 1)

        # Decrease exposure on 'd' key press.
        elif _key == ord('d'):
            # Get current exposure.
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            # Decrease exposure by 1.
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure - 1)

colors = {0: (0, 255, 255),
          47: (0,0,255)  # Red for class 2             
          }

# Create a counter
frame_counter = 0

# Initialize the deque to store the last 100 predicted sugar contents
predicted_sugar_contents = deque(maxlen=100)

# Loop through the video frames
while cap.isOpened():

    start_time = time.time()
    prev_time = start_time

    # Read a frame from the video
    ret, frame      = cap.read()
    
    h, w, c = frame.shape
    


    if ret == True: # Run YOLOv8 inference on the frame
        # Change the frame size
        results = model(frame, imgsz=224)
        result = results[0]
        len_result = len(result)

        cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

        if len_result == 0:
            # when no object in frame
            cv2.rectangle(frame, (0,0,w,h), (153,255,255), 7)
            cv2.putText(frame, f'Put the Apple', (230, 200), user_font, 0.7, (153,255,255), 4)
            cv2.putText(frame, f'Put the Apple', (230, 200), user_font, 0.7, (255,255,255), 2)

        for idx in range(len_result):
            
            detection = result[idx]
                    
            box = detection.boxes.cpu().numpy()[0]
            cls = int(box.cls[0])
            conf = box.conf[0]

            diff_time = time.time() - prev_time
            
            if diff_time > 0:
                fps = 1 / diff_time

            cv2.putText(frame, f'FPS : {fps:.2f}', (20, 50), user_font, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, f'FPS : {fps:.2f}', (20, 50), user_font, 0.7, (255, 255, 255), 2)
            
            # 'Enter' Key: predict Brix
            key = cv2.waitKey(1) & 0xFF
            if key == ord('\r'): signal = 1
            # 'r' Key: Stop predict Brix
            elif key == ord('r'): signal = 0

            # When stop predict Brix
            if signal == 0:
                predicted_sugar_contents = []
                cv2.rectangle(frame, (0,0,w,h), (255,102,102), 7)
                cv2.putText(frame, f'Press Enter for Check the Brix', (130, 200), user_font, 0.7, (255,102,102), 4)
                cv2.putText(frame, f'Press Enter for Check the Brix', (130, 200), user_font, 0.7, (255,255,255), 2)

            # When predict Brix
            elif signal == 1:
                iteration = 0
                cv2.rectangle(frame, (0,0,w,h), (102,102,255), 7)
                cv2.putText(frame, f'Press r key for end check', (210, 400), user_font, 0.5, (102,102,255), 3)
                cv2.putText(frame, f'Press r key for end check', (210, 400), user_font, 0.5, (255,255,255), 2)

                # if Apple confidence over 0.7
                if cls == 47 and conf > 0.7:
                    iteration += 1

                    segment = detection.masks.xy[0].astype(np.int32)
                    print("detection.masks",segment)
                                
                    color = colors[cls]
                    if segment.shape[0] > 0:
                        
                        # Use only apple regions for prediction
                        mask_base = np.zeros(frame.shape, dtype = np.uint8)
                        cv2.fillPoly(mask_base, [segment], (255,255,255))
                        img_gray = cv2.cvtColor(mask_base, cv2.COLOR_BGR2GRAY)
                        mask2 = cv2.bitwise_and(frame, frame, mask = img_gray)
                        
                        # Get bounding box coordinates
                        min_x, min_y = np.min(segment, axis=0)
                        max_x, max_y = np.max(segment, axis=0)
                        
                        # Draw the bounding rectangle
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (102,102,255), 3)  # you can change the color and thickness
      
                        # In your main loop, after predicting the sugar content, append it to the deque
                        predicted_sugar_content = round(predict_sugar_content(mask2), 2)  
                        predicted_sugar_contents.append(predicted_sugar_content)

                        # Calculate the average predicted sugar content from the values in the deque
                        avg_predicted_sugar_content = sum(predicted_sugar_contents) / len(predicted_sugar_contents)
                       
                        cv2.putText(frame, f"Brix: {avg_predicted_sugar_content:.2f}", (min_x+70, min_y+100), user_font, 0.5, (102,102,255), 2)
                        cv2.putText(frame, f"Brix: {avg_predicted_sugar_content:.2f}", (min_x+70, min_y+100), user_font, 0.5, (255,255,255), 1)
                    
                    
             

        cv2.imshow("video", frame)

        out.write(frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == 27   :   break

    else:
        print("Camera is Disconnected ...!")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
```
