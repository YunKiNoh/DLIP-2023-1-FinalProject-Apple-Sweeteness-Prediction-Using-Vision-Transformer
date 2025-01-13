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