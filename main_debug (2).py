import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import yaml
import pyttsx3
from openvino.runtime import Core 
from utils.augmentations import letterbox
with open('data/coco.yaml') as f:
   #d = yaml.safe_load(f)  # dict
   result = yaml.load(f.read(),Loader = yaml.FullLoader)
class_list = result['names']
#print(class_list)
 #  for i, x in enumerate(d['names']):
 #    print(i, x)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
#model = torch.hub.load('ultralytics/yolov5', 'yolov5l6.xml')
#model.conf = 0.5
#model  = "yolov5l6.xml"
model  = "yolov8n.xml"
ie=Core()
net=ie.compile_model(model=model,device_name="AUTO")


def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    #print("box is XXXXX")
    #print(box)
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置,通过取边界框的x和y坐标的平均值来得到,(//向下取整)
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #计算一个随机的偏移量bias，该偏移量范围为-min_val // 4到min_val // 4，通过将这个偏移量应用到mid_pos计算得到一个新的位置，然后从depth_data数组中获取对应位置的值dist
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波，得到一个可靠的值
    d = np.mean(distance_list)
    m = (box[0] + box[2])//2
    #if d >= 300 and d <= 10000:
    #	#print(distance_list, np.mean(distance_list))
    #	print(box[-1])
    #	print(d)
    #	pyttsx3.speak(box[-1])
    #	pyttsx3.speak('is')
    #	pyttsx3.speak(round(d/100)/100)
    #	pyttsx3.speak("meters")
    #	pyttsx3.speak("on your")
    #	if m <= 325:
    #	    pyttsx3.speak("right")
    #	else:
    #	    pyttsx3.speak("left")
    return d #计算返回数组的平均值


def dectshow(org_img, boxs,depth_data):
    img = org_img.copy()
    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        dist = get_mid_pos(org_img, box, depth_data, 24)
        cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('dec_img', img)

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    profile=pipeline.start(config)
    device = profile.get_device()
    device.hardware_reset()
    output_node=net.outputs[0]
    #ir=net.creat_infer_request()
    cap=cv2.VideoCapture(0)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            
            start = time.time()
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            
            

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())
            
            inputImage, ratio, (dw,dh) = letterbox(color_image,auto=False)

            #results = model(color_image)
            blob=cv2.dnn.blobFromImage(color_image,1 / 255.0, (640,640),swapRB=True,crop=False)
            outputs=net([blob])[net.outputs[0]]
            #outputs=ir.infer(blob)[output_node]
            outputs=np.array([cv2.transpose(outputs[0])])
            rows=outputs.shape[1]
            end=time.time()
            sum_time=end-start
            fps=1/sum_time
            print(fps)
            #print(results)
            class_ids=[]
            scores=[]
            boxes=[]
            boxs=[]
            for i in range(rows):
            	classes_scores=outputs[0][i][4:]
            	(minScore,maxScore,minClassLoc,(x,maxClassIndex))=cv2.minMaxLoc(classes_scores)
            	if maxScore>=0.25:
            		class_ids.append(maxClassIndex)
            		box = [outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),outputs[0][i][2],outputs[0][i][3]]
            		scores.append(maxScore)
            		boxes.append(box)
            indexes = cv2.dnn.NMSBoxes(boxes,scores,0.5,0.6)
            	
            for i in indexes:
            	finalbox=[float(boxes[i][0]),float(boxes[i][1]),float(boxes[i][2]),float(boxes[i][3]),scores[i],class_ids[i],class_list[class_ids[i]]]
            	#print("final bx is XXXXXXX")
            	print(i)
            	boxs.append(finalbox)
            			
            			
            
            #boxs= results.pandas().xyxy[0].values
            #boxs = np.load('temp.npy',allow_pickle=True)
            resized_depth_img = cv2.resize(depth_image, (640, 640))
            dectshow(color_image, boxs, resized_depth_img)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
