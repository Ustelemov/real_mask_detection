#!/usr/bin/env python3
import os
import cv2
import sys
import glob 
import argparse
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config,Model
from hyperpose.Model.common import image_float_to_uint8
import time
from sort import Sort
import uuid
import onnxruntime as ort
import json
import yaml


### Model start

hyperpose_model_type = "MobilenetThinOpenpose"
#available options: Mobilenet, Vggtiny, Vgg19, Resnet18, Resnet50")
hyperpose_model_backbone = "MobilenetThin"
hyperpose_model_name = "default_name"
# hyperpose_weights_path = "hyperpose/Weights/lightweight_openpose.npz"
hyperpose_weights_path = "hyperpose/Weights/lightweight_openpose_vggtiny.npz"

# config model
Config.set_model_name(hyperpose_model_name)
Config.set_model_type(Config.MODEL.LightweightOpenpose)
# Config.set_model_type(Config.MODEL.Openpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# Config.set_model_backbone(Config.BACKBONE.Default)
config = Config.get_config()


# contruct model and processors
model = Model.get_model(config)
# post processor
PostProcessorClass = Model.get_postprocessor(config)
post_processor = PostProcessorClass(parts=model.parts, limbs=model.limbs, hin=model.hin, win=model.win, hout=model.hout,
                                wout=model.wout, colors=model.colors)
# image processor
ImageProcessorClass = Model.get_imageprocessor()

# load weights
model.load_weights(hyperpose_weights_path, format="npz_dict")
model.eval()

# onnx model
onnx_model = "onnx_models/TinyVGG-V2-HW=342x368.onnx"
onnx_model_width = 368
onnx_model_height = 342

image_processor = ImageProcessorClass(input_h=onnx_model_height, input_w=onnx_model_width)

so = ort.SessionOptions()
# so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# so.intra_op_num_threads = 4
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
# so.inter_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


onnx_model_session = ort.InferenceSession(onnx_model, sess_options=so)
# onnx_model_session.set_providers(['CUDAExecutionProvider'])
onnx_input_name = onnx_model_session.get_inputs()[0].name
onnx_output_name_0 = onnx_model_session.get_outputs()[0].name
onnx_output_name_1 = onnx_model_session.get_outputs()[1].name

### Model end

def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

class FaceData:
    def __init__(self):
        self.faces = dict()
        self.path = "./output/saved_faces"
        self.detections_count_threshold = 15
        self.detections_score_threshold = 0.80

    def add(self, id, arr, score):
        if id not in self.faces:
            self.faces[id] = []
        self.faces[id].append((arr,score))

    def remove(self, id):
        if id not in self.faces:
            return
        
        dataArr = self.faces[id]

        #remove element
        del self.faces[id]

        if len(dataArr) < self.detections_count_threshold:
            return

        score = 0
        scores = []
        img = []

        for data in dataArr[-self.detections_count_threshold:]:
            scores.append(data[1])
            if data[1] > score:
                score = data[1]
                img = data[0]
        

        if score < self.detections_score_threshold:
            return


        mkdir(self.path)

        h,w,c = img.shape
        id = str(uuid.uuid1())

        cv2.imwrite("{0}/{1}_{2}_{3}_{4}.jpg".format(self.path, id, time.time(), w,h), img)

        img_64 = cv2.resize(img, (64,64))
        cv2.imwrite("{0}/{1}_{2}_64_64.jpg".format(self.path, id, time.time()), img_64)

        img_96 = cv2.resize(img, (96,96))
        cv2.imwrite("{0}/{1}_{2}_96_96.jpg".format(self.path, id, time.time()), img_96)

        img_128 = cv2.resize(img, (128,128))
        cv2.imwrite("{0}/{1}_{2}_128_128.jpg".format(self.path, id, time.time()), img_128)

        img_160 = cv2.resize(img, (160,160))
        cv2.imwrite("{0}/{1}_{2}_160_160.jpg".format(self.path, id, time.time()), img_160)

        img_196 = cv2.resize(img, (196,196))
        cv2.imwrite("{0}/{1}_{2}_196_196.jpg".format(self.path, id, time.time()), img_196)

        img_224 = cv2.resize(img, (224,224))
        cv2.imwrite("{0}/{1}_{2}_224_224.jpg".format(self.path, id, time.time()), img_224)


def process_image(image, humans, name):
    # result_image = image
    result_image = image_float_to_uint8(image.copy())
    faces = []
    for human in humans:

        if human.get_score() < 0.1:
            break

        result_image = human.draw_human(result_image)

        ok, top_left, bottom_right = human.get_head_bboxes(image)

        if not ok:
            continue

        score = human.get_head_score()
        distance_h = bottom_right[1] - top_left[1] 
        distance_w = bottom_right[0] - top_left[0] 

        if distance_h > 10 and distance_w > 10:
            faces.append([top_left[0], top_left[1], bottom_right[0], bottom_right[1], score])

    trackers, removed = tracker.update(np.array(faces))
    
    # visualize and add
    for d in trackers:
        score = d[5]
        d = d.astype(np.int32)

        cv2.rectangle(result_image, (d[0], d[1]), (d[2], d[3]), (0,0,0), 3)
        cv2.putText(result_image, 'ID%d (%0.1f)' % (d[4], score), 
            (d[0] - 5, d[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)

        # add
        face = image[d[1]:d[3], d[0]:d[2]]
        face_data.add(d[4],face, score)

    # removed
    for rm in removed:
        face_data.remove(rm)

    return result_image

if __name__ == '__main__':
    mkdir("output")

    tracker = Sort(max_age=12, min_hits=0) 
    face_data = FaceData()


    cap = cv2.VideoCapture('./input/input.mp4')
    length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output/output.mp4', fourcc, 30.0, (400,400))


    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret:

            frame_count = frame_count + 1
            print(frame_count," of ", length)


            # input video already in (1280,720)
            frame = cv2.resize(frame, (1280,720))

            # get only entrance from image
            frame = frame[:400,150:550]

            # img_res = cv2.resize(frame, (onnx_model_width,onnx_model_height))
            # img_res.resize((1,3,onnx_model_height,onnx_model_width))

            # data = json.dumps({'data':img_res.tolist()})
            # data = np.array(json.loads(data)['data']).astype('float32')


            image = image_processor.read_image_rgb_float(frame)
            input_image, scale, pad = image_processor.image_pad_and_scale(image)
            input_image = np.transpose(input_image,[2,0,1])[np.newaxis,:,:,:]

            #model forward
            # predict_x = model.forward(input_image)
            data = json.dumps({'data':input_image.tolist()})
            data = np.array(json.loads(data)['data']).astype('float32')
            start = time.time()
            conf_map, paf_map, conf_map1, paf_map1 = onnx_model_session.run([onnx_output_name_0, onnx_output_name_1, onnx_output_name_0, onnx_output_name_1], {onnx_input_name: data, onnx_input_name: data})

            predict_x = dict()
            predict_x['conf_map'] = conf_map
            predict_x['paf_map'] = paf_map
            # post process
            humans = post_processor.process(predict_x)[0]
            print("time: ", time.time()-start)
            # visualize results (restore detected humans)
            print(f"{len(humans)} humans detected")
            for human_idx,human in enumerate(humans,start=1):
                human.unpad(pad)
                human.unscale(scale)
            
            frame = process_image(image=frame, humans=humans, name="result")

            # cv2.imshow('Video', frame)
            out.write(frame)

            key = cv2.waitKey(1) & 0xFF

            # If the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
