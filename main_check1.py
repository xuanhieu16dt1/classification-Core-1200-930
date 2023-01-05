import argparse
import time
import tkinter
from tkinter import Tk, Button

import cv2
import numpy as np
import tensorflow as tf
from classes import TensorflowLiteClassificationModel

# from tkinter.constants import TRUE     # from tkinter import Tk for Python 3.x

ap = argparse.ArgumentParser()
from PIL import ImageTk
from PIL import Image

class_names = ['930', '1200']
ap.add_argument('-c', '--config', default="weights/yolov3-tiny.cfg",
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default="weights/yolov3-tiny_20000.weights",
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default="weights/yolo.names",
                help='path to text file containing class names')
args = ap.parse_args()
model_detection = cv2.dnn.readNet(args.weights, args.config)
model_classification = TensorflowLiteClassificationModel("weights/model_quant_tl.tflite")
dispW = 3264
dispH = 2464
flip = 2
camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=' + str(
            flip) + ' ! video/x-raw, width=' + str(dispW) + ', height=' + str(
            dispH) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.3 brightness=-.2 saturation=1.2 ! appsink'
cap = cv2.VideoCapture(camSet)


def get_output_layers(model_detection):
    layer_names = model_detection.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model_detection.getUnconnectedOutLayers()]
    return output_layers


def classify_object(image, image_size=(120, 120)):
    image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_array = np.asarray(image).astype(np.float32)
    img_array = tf.expand_dims(img_array, 0)
    label_id = model_classification.run(img_array)
    return label_id


def detect_object(image, model_detection):
    image_view = image.copy()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)
    model_detection.setInput(blob)
    outs = model_detection.forward(get_output_layers(model_detection))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.9
    nms_threshold = 0.5

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    indices = sorted([idc[0] for idc in indices])
    return np.array(boxes)[indices]


def detect(image, model_detection, model_classification):
    t1 = time.time()
    image_view = image.copy()
    count_id = [0, 0]
    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    boxes = detect_object(image, model_detection)
    predict_results = []
    for index, box in enumerate(boxes):
        # mở rộng box trước khi predict classify model
        x = box[0] - int(box[2] * 0.1)
        y = box[1] - int(box[3] * 0.1)
        w = box[2] + int(box[2] * 0.2)
        h = box[3] + int(box[3] * 0.2)
        img_crop = image_view[y:y + h, x:x + w, :]
        if img_crop.shape[0] * img_crop.shape[1] != 0:
            # cv2.imwrite(f"./object_detected/object_{index}.jpg", img_crop)
            label_id = classify_object(image=img_crop)[0]
            print(label_id)
            predict_results.append((box, label_id))
            # cv2.imwrite(f"{detected_path[label_id]}/crop_{count_object}.jpg", img_crop)
            count_id[label_id] += 1
            # image_view = cv2.rectangle(image_view, (x, y), (x + w, y + h), colors[label_id], 3)
        colors = [(0, 255, 0), (0, 0, 255)] if count_id[0] > count_id[1] else [(0, 0, 255), (0, 255, 0)]
        for result in predict_results:
            cv2.putText(image_view, class_names[result[1]], (result[0][0], result[0][1]), 0, 1, colors[result[1]], 4)
    t2 = time.time()
    t = t2 - t1
    cv2.putText(image_view, f"{class_names[0]}: {count_id[0]} pcs.", (0, 80), 0, 3, (0, 0, 0), 3)
    cv2.putText(image_view, f"{class_names[1]}: {count_id[1]} pcs.", (0, 170), 0, 3, (0, 0, 0), 3)
    cv2.putText(image_view, "Process Time :" + f"{t:0.02}" + "s", (0, 260), 0, 3, (0, 0, 0), 3)
    cv2.putText(image_view, f"{class_names[0]}: {count_id[0]} pcs.", (2, 82), 0, 3, (255, 255, 255), 3)
    cv2.putText(image_view, f"{class_names[1]}: {count_id[1]} pcs.", (2, 172), 0, 3, (255, 255, 255), 3)
    cv2.putText(image_view, "Process Time :" + f"{t:0.02}" + "s", (2, 262), 0, 3, (255, 255, 255), 3)
    return image_view


def main():
    window = Tk()
    window.geometry("1480x980")
    window.title("Phân Loại Core")
    window.configure(background="#E1DDDC")
    image2 = Image.open(r"C:\Users\xuanhieuktsx\YOLOv3\logo2.jpg")
    image3 = Image.open(r"C:\Users\xuanhieuktsx\YOLOv3\logo3.jpg")
    test2 = ImageTk.PhotoImage(image2)
    test3 = ImageTk.PhotoImage(image3)
    label2 = tkinter.Label(image=test2)
    label3 = tkinter.Label(image=test3)
    label2.image = test2
    label2.place(x=1280, y=0)
    label3.image = test3
    label3.place(x=0)

    def on_click():
        #t1 = time.time()
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            print("cannot open camera ")

        # ret, frame = cap.read() #stock
        ret, frame = True, cv2.imread("1200_1.jpg")
        result = detect(frame, model_detection, model_classification)
        result = cv2.resize(result, (980, 840))
        blue, green, red = cv2.split(result)
        test = ImageTk.PhotoImage(Image.fromarray(cv2.merge((red, green, blue))))
        label1 = tkinter.Label(image=test)
        label1.image = test
        label1.place(x=240, y=70)

    btn2 = Button(window, text="Exit", command=lambda: exit(), bg="green", fg="white", padx=30, pady=20)
    btn2.place(x=1360, y=860)
    btn3 = Button(window, text="SHOW CAM", command=lambda: [f() for f in [on_click]], bg="green", fg="black",
                  padx=60, pady=20)
    btn3.place(x=600, y=0)
    window.mainloop()


if __name__ == "__main__":
    main()
