# Apple Sweeteness Prediction Using Vision Transformer

 

**date**: 2023/06/19

**Author**: Yun Ki Noh/ EunChan Kim

**Github**: https://github.com/GracenPraise/DLIP2023

**Demo Video**: https://youtu.be/GrVAf8BUc9s

## 1. Introduction

이 프로젝트는 여러 사과가 섞여 있고, 어떤 사과가 더 달콤한지 알 수 없는 상황에서 고객이 사과를 쉽게 고를 수 있도록 돕는 것을 목표로 하고 있습니다. 사람들은 돈을 효율적으로 쓰고 싶어 하기 때문에 사과를 고르는 데 더 많은 시간을 쓸 가능성이 높습니다. 사과 당도를 확인하는 방법이 몇 가지 있긴 하지만, 주스를 짜거나 적외선을 사용하는 방법은 사과를 망가뜨리거나 비용이 많이 드는 문제가 있습니다. 그래서 사과에 손상을 주지 않으면서도 비싸지 않은 방법으로 브릭스(당도)를 예측하기 위해 딥러닝 모델을 사용하기로 결정했습니다.

### Goal

사과의 당도(브릭스)를 예측하기 위해 딥러닝 모델을 훈련함으로써, 사과의 당도를 실시간으로 예측하고 그 결과를 출력하는 것이 가능합니다. 사과의 당도를 예측하는 선행 연구가 없었기 때문에, 훈련 손실(train loss)은 MSE(Mean Squared Error) 기준 2 이내로 설정하였으며, 실제 당도 값과 예측 값 간의 오차는 임의로 ±5% 이내로 설정하였습니다.

### Hardware

* NVIDIA GeForce RTX 3050

* S604HD VER 1.0

* Server Computer(DSTECH)

<div align="center">

| Device | Specification                        |
|--------|--------------------------------------|
| GPU    | 1xAMD EPYC 7742 2.25GHz Upto 3.4GHz |
| CPU    | 4EA x NVIDIA A100 80GB (320GB VRAM) |

</div>

<div align="center">
  Table 1. Server Computer</p>
</div>


#### Hardware setting

<div align="center">
  <img width="940" alt="angle_picture" src="https://github.com/YunKiNoh/DLIP-2023-1-FinalProject-Apple-Sweeteness-Prediction-Using-Vision-Transformer/blob/main/image/angle_picture.png" /><br>
  <p style="margin-top: 10px;">
</div>

<div align="center">
  <img width="940" alt="angle" src="https://github.com/YunKiNoh/DLIP-2023-1-FinalProject-Apple-Sweeteness-Prediction-Using-Vision-Transformer/blob/main/image/angle.png" /><br>
  <p style="margin-top: 10px;">Figure 1. Camera's Angle & Distance</p>
</div>


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

<div align="center">
  <img width="940" alt="dataset" src="https://github.com/YunKiNoh/DLIP-2023-1-FinalProject-Apple-Sweeteness-Prediction-Using-Vision-Transformer/blob/main/image/dataset.png" /><br>
  <p style="margin-top: 10px;">Figure 2. Deep Learning Dataset about apple from Jeonbuk</p>
</div>

저희는 딥러닝 모델을 훈련하기 위해 전북시에서 제공하는 사과 데이터셋을 사용하였습니다. 한국에서 가장 많이 소비되는 후지(Fuji) 사과를 당도 예측 대상으로 선정하였고, 딥러닝 학습을 수행하기 위해 후지 사과의 17,000개의 이미지와 라벨링된 데이터를 사용하였습니다.



### Format of dataset

Dataset from Jeonbuk is provided labeling data as json file. 

<div align="center">

|      Sugar Content       |         Segmentation         |
| :----------------------: | :--------------------------: |
| <img width="195" alt="labeling" src="https://github.com/YunKiNoh/DLIP-2023-1-FinalProject-Apple-Sweeteness-Prediction-Using-Vision-Transformer/blob/main/image/labeling.png"> | <img width="468" alt="segmentation" src="https://github.com/YunKiNoh/DLIP-2023-1-FinalProject-Apple-Sweeteness-Prediction-Using-Vision-Transformer/blob/main/image/segmentation.png"> |

<br>
<p>Table 2. Labeling Data</p>

</div>


라벨링된 데이터셋에서는 '분할 좌표(segmentation coordination)'를 사용하여 사과의 픽셀을 추출하고, '당도 정보(sugar content's information)'를 활용하여 파이썬 코드로 당도를 예측하였습니다. 

**Dataset link**: [AI-Hub ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=490)



## 3. Tutorial Procedure

### Setup & Installation

파이썬에서 딥러닝 모델을 훈련하기에 앞서 몇 가지 준비 설정이 필요합니다. 

#### 1. Anaconda Virtual Environment

해당 프로젝트를 위해 개별적으로 파이썬 환경을 구축하기 위해서는 가상 환경을 생성하고 이곳에 여러가지 라이브러리를 설치해야 합니다. 우선 가상 환경을 생성하기 위해 Anaconda에서 다음 코드를 입력합니다.

``` 
conda create -n py39 python=3.9.13
```

만약 가상환경을 활성화하고 싶다면 다음을 실행합니다.

``` 
conda activate py39
```

#### 2. Install Libs 

다음과 같은 라이브러리들을 가상환걍에 설치합니다. [Numpy, OpenCV, Matplot, Jupyter].

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

Visual Studio Code(VS Code)를 설치합니다. 

**link**: [VS Code Downloads](https://code.visualstudio.com/download)

#### 4-1. Server Computer's CUDA, cuDNN(Option) 

가상환경에 CUDA와 PyTorch를 설치합니다. 이번 프로젝트에서 사용한 버전은 다음과 같습니다.

* CUDA 11.3.1
* PyTorch 1.10

#### 4-2. Install GPU Driver, CUDA, cuDNN

Because each driver has proper versions for compatbility, you must select the appropriate version of GPU Driver, CUDA, cuDNN.
개인 노트북마다 호환되는 드라이버 버전이 다르기 때문에 개인 노트북의 사양을 확인하여  GPU Driver, CUDA, 그리고 cuDNN을 설치합니다. 자신에게 맞는 버전은 다음을 통해서 확인합니다.

**1. Check PyTorch & CUDA Version**

<img width="635" alt="pytorch" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/3595a722-11ec-420f-b1a8-d94f590116bb">

> **Figure 3. Pytorch & CUDA Version**

PyTorch를 사용하기 위해서는 11.7 version 또는 11.8 version의 CUDA를 사용해야 합니다.

**2. Check Proper GPU Version for CUDA 11.7** 

<img width="733" alt="toolkit" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/bdc69e89-e0d0-40db-bb62-7f93d227200b">

> **Figure 4. CUDA Toolkit**

**Install GPU Driver**

<img width="387" alt="gpudriver" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/133fffb1-938d-41b8-9d89-66fa70ddd937">

> **Figure 5. GPU Driver**

GPU 드라이버의 경우 다음을 통해서 노트북 사양에 맞는 소프트웨어 버전을 찾은 뒤 설치해줍니다. 

**link**:[GPU Driver](https://www.nvidia.co.kr/Download/index.aspx?lang=kr)

**3. Check Proper cuDNN Version for CUDA 11.7** 

<img width="1024" alt="cudnn" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/6a12ba30-1f3d-4380-ac23-cc2d8e04a1e3">

> **Figure 6. cuDNN version**

**link**:[cuDNN version](https://developer.nvidia.com/rdp/cudnn-archive)

**4. Install CUDA**

Install CUDA 11.7

**link**:[CUDA Toolkit 11.7 Downloads](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

**5. Install PyTorch**

PyTorch를 CUDA 11.7 version과 함께 설치합니다.

**link**:[PyTorch Downloads](https://pytorch.org/)

**6. Install cuDNN**

**link**:[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

## 4. Train & Test Process

### Train  Process

#### 1. Apple Segentation

딥러닝을 통해서 사과의 당도를 실시간으로 예측하기 위해서는 이미지 전체가 아닌 사과 이미지 내의 정보만을 가져오는 작업이 필요합니다. 이는 segmentation(인식한 물체의 이미지 정보를 픽셀 단위로 들고오는 작업)을 통해서 수행할 수 있습니다. segmentation의 경우 YOLO의 딥러닝 모델 중 상위 모델에서 제공하고 있는 기술인데, Yolov8s-seg에서 제공하는 사전 학습된 모델이 사과 세그멘테이션에 가장 우수한 성능을 보여 이 모델을 사과를 인식하고 해당 사과의 색깔 정보를 들고 오는데 활용하였습니다.

<img width="486" alt="seg" src="https://github.com/GracenPraise/DLIP2023/assets/91367451/5f51f480-090c-4774-ad52-19a6be9272cf">

> **Figure 7. Result of apple segmentation with yolov8-seg**

#### 2. Brix Prediction

본 프로젝트에서는 사과의 당도를 예측하기 위해서 사과의 이미지와 당도 값만을 사용하기로 결정하였으며, 이를 위해서는 사과의 색깔 정보와 당도 사이의 관계를 학습할 수 있는 회귀 모델이 필요하였습니다. 따라서 정교하고 높은 정확도를 가진 모델을 사용하는 것이 적합하다고 판단하였으며, 그 결과 ResNet50, DenseNet, VGG16, Inception v3 등 다양한 모델을 실험해보았습니다. 다만 기대에 비해 정확도가 높이 않음을 확인할 수 있었습니다.

![models](https://github.com/GracenPraise/DLIP2023/assets/91367451/230886fc-1a29-49c1-902d-dfedb5d6f91f)

> **Figure 8. MES loss comparison of multiple deep learning model**

따라서, 최근 이미지 처리에서 딥러닝에 널리 활용되고 있는 Vision Transformer(ViT) 모델을 사용하기로 결정하였습니다. 구체적으로는, 입력 크기가 224x224이고 16-패치(patch) 방식을 사용하는 vit_base_patch16_224 모델을 채택하였습니다.

![vit](https://github.com/GracenPraise/DLIP2023/assets/91367451/a96e86d7-e83c-4e67-9865-0d195376a184)

> **Figure 8. Structure of Vision Transformers (VIT). Referenced by 'Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.'**

제공된 사과 데이터셋에서 일부 사진에 회전이 적용되어 세그멘테이션 좌표가 올바르게 맞지 않는 문제가 있었습니다. 이 문제를 해결하기 위해 사진의 EXIF 정보를 추출하여 회전이 적용된 경우 이를 조정하는 전처리 과정을 수행하여 올바른 정렬을 보장하였습니다.

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



훈련에는 약 170,000장의 이미지를 사용하였고, 검증에는 약 17,000장의 이미지를 활용하였습니다. 훈련 이미지의 수가 많아 에포크(epoch) 크기를 늘렸으나, 이로 인해 훈련 시간이 크게 증가하는 문제가 발생하였습니다. 5에포크 이후에는 손실값이 크게 감소하지 않는 것을 관찰한 후, 최종적으로 에포크 수를 10으로 설정하였습니다. 손실 측정을 위해 회귀 문제에서 흔히 사용되는 평균 제곱 오차(Mean Squared Error, MSE) 손실 함수를 적용하였으며, 옵티마이저는 Adam을 사용하였습니다.

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

프로젝트 진행 중 설정된 목표 훈련 손실값인 2 이하를 달성한 것을 확인하였으며, 훈련 과정에서 과적합의 징후 없이 원활하게 진행되었음을 확인하였습니다.

![loss](https://github.com/GracenPraise/DLIP2023/assets/91367451/79ed37bb-f551-4ecc-baa8-2e9b5494548a)

> **Graph1. Train and Validation Loss Graph**



### Test Process

**Model Download: **[link](https://drive.google.com/drive/folders/12fvvtiPWf_keCtu5OOUQ4VUQA91ZJuhU?usp=drive_link)

위에서 개발한 세그멘테이션 모델과 회귀 모델을 활용하여 사과의 당도를 실시간으로 예측할 수 있도록 코드를 작성하였습니다. 간단한 흐름은 다음과 같습니다.

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

먼저, yolov8-seg 모델을 로드하여 사과를 감지합니다. 만약 사과가 감지되지 않으면 'Put the Apple' 메시지가 표시됩니다.

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

객체가 감지되고 Enter 키가 눌리면 세그멘테이션 과정이 시작됩니다. 이 과정에서 마스크 연산이 수행되어 사과 영역 내의 데이터만 분리됩니다. 이후 처리와 분석은 세그멘테이션된 사과 영역 내의 데이터만을 사용하여 진행됩니다.

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

세그멘테이션된 영역 내에서 얻어진 데이터는 사과 당도의 예측에 사용됩니다. 예측된 당도 값은 화면에 표시됩니다.

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

최종적으로 결과가 화면에 표시됩니다. 'r' 키를 누르면 이전 단계인 세그멘테이션 단계로 돌아갈 수 있습니다.

실시간으로 사과 당도를 예측하는 모델의 높은 연산 부담으로 인해 프레임 드롭이 발생하는 문제가 있습니다. 이를 완화하기 위해, 사과가 자리에 놓이고 Enter 키가 눌렸을 때만 예측을 수행하도록 설정하였습니다. 또한, 각 프레임 변화에 따른 값의 변동을 방지하기 위해 여러 프레임에서 예측된 당도 값을 평균으로 계산합니다. 예측된 당도 값은 계속 누적되어 평균값이 계산되며, 사과가 바뀌는 경우 'r' 키를 눌러 누적된 값을 초기화한 뒤 Enter 키를 다시 눌러 예측을 진행해야 합니다.

| ![ready](https://github.com/GracenPraise/DLIP2023/assets/91367451/ba8f6e11-ec82-4565-bc6d-8b688a37ba02) | ![press_enter](https://github.com/GracenPraise/DLIP2023/assets/91367451/2fb3adb9-62b7-492c-8600-97b9e51f096f) | ![result](https://github.com/GracenPraise/DLIP2023/assets/91367451/0f0d6a77-ac59-4752-9504-66418f435a9b) |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
|  **Figure 9. When No Object in Frame**   |    **Figure 10. When Apple Deteted**     |      **Figure 11. Brix Prediction**      |



## 5. Results and Analysis

### Results

![result_brix](https://github.com/GracenPraise/DLIP2023/assets/91367451/7102e510-d952-4672-9265-2948cf7be08f)

> **Figure 12. Result**

저희는 이전의 과정들을 통해서 카메라를 통해 실시간으로 캡쳐하고 있는 사과의 당도를 예측할 수 있었습니다. 다만 해당 값이 정확한지 확인하기 위해서 다음과 같이 평가 기준을 세우고 정확도를 평가하였습니다.

훈련된 모델의 신뢰성을 확인하기 위해 사과 10개를 선택하였으며, 사과들의 실제 브릭스 값을 즙을 통하여 측정하였고, 모델이 예측한 값이 직접 측정한 당도와 얼마나 차이가 있는지 확인하였습니다. 그 결과는 다음과 같습니다.

|                  |   1   |   2   |   3   |   4   |   5   |   6   |   7   |  8   |   9   |  10   |
| :--------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: | :---: | :---: |
| Real[brix]       | 12.5  | 13.0  | 13.7  | 11.0  | 12.0  | 10.0  | 14.2  | 11.0 | 13.0  | 12.0  |
| Prediction[birx] | 12.44 | 12.76 | 13.15 | 11.71 | 12.16 | 11.53 | 14.12 | 11.7 | 12.88 | 13.63 |
| Error[%]         | 0.48  | 1.85  | 4.01  | 6.45  | 1.33  | 15.3  | 0.56  | 6.36 | 0.92  | 13.58 |

| Total[%] |
| :------: |
|   5.08   |

일부 사과에 대해서 급격히 오차가 증가하며 오차를 5% 이내의 정확도를 달성한다는 목표는 달성하지 못하였습니다.

> **Table 3. Evaluation Error**

|             | Train Error[MSE] | Validation Error[MSE] | Evaluation Error[%] |
| :---------: | :--------------: | :-------------------: | :-----------------: |
|    Goal     |     Under 2      |    No Overfitting     |      Under 5%       |
|   Result    |       1.89       |         3.02          |        5.08         |
| Achievement |        O         |           O           |          X          |

> **Table 4. Result**

모델 훈련 과정에서 훈련 손실이 2 MSE 이하로 유지되는 것을 확인하였고, 검증 오류를 통해 모델이 과적합되지 않았음을 확인할 수 있었습니다. 그러나 평가 목표를 달성하는 데에는 실패하였습니다.



### Analysis

#### Light condition

훈련에 사용된 데이터를 검토한 결과, 70% 이상의 이미지가 과수원에서 촬영된 것으로 나타났습니다. 이는 충분한 조명 조건에서 촬영된 사진이 대부분임을 의미하며, 실내 환경에서 훈련된 모델을 평가할 경우 오차가 발생할 수 있음을 시사합니다. 게다가 이번 프로젝트에서는 사과의 당도를 예측하기 위해 오로지 색깔 정보만을 의지하였기 때문에 이러한 광학 환경의 변화가 당도 예측에 더욱 큰 영향을 주는 것으로 보입니다. 이를 통해서 저희는 이미 시중에서 활용되고 있는 과일 당도 예측 사례들이 왜 외부 빛을 차단한 제한된 공간에서 주로 이루어지고 있는 지를 알 수 있었습니다. 실제로 실험했던 장소가 아닌 다른 곳에서 시연했을 때 당도 예측의 오차율이 급증했는데, 당도 예측의 안정성 및 정확도를 높이기 위해서는 제한된 환경에서 예측이 이루어져야 할 것입니다.

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
