#!/usr/bin/env python
import cv2
import os
import sys
import init_path
from project302 import Project302
import matplotlib.pyplot as plt 
import config as cfg
import caffe 
import numpy as np
import time 

#img_dir ='/data2/detection/py-R-FCN-171219/data/capture302/1_pic'; 
#img_names = os.listdir(img_dir);

# This is ur testing video path
'''
Here we initialize the project302
1. init the class 
2. set detector model 
3. set tracker model
4. set nms and confidence threshold
4. surveillance   
''' 

project = Project302(1, 8, do_verify=True)

project.init_detector(cfg.detector_mtcnn)
project.init_verifier('../models/verifier/verifier.prototxt', '../models/verifier/verifier.caffemodel')

#project.init_tracker(cfg.tracker_goturn+'.prototxt',cfg.tracker_goturn+ '.caffemodel');
#print('init tracker success');
#project.init_detector(cfg.detctor_rfcn+'.prototxt',cfg.detctor_rfcn + '.caffemodel');
#project.SetNMS(0.7);

print('dalong log : init success');
# initialize the video input and output 
frame_width = 640;
frame_height = 480;

# SET CAFFE
if (cfg.GPU_MODE):
	caffe.set_mode_gpu()
	caffe.set_device(0)
else:
	caffe.set_mode_cpu()

def show_result(image,dets):
	#image = image[:,:,(2,1,0)];
	if(np.size(dets) == 0):
		return image
	for index in xrange(np.size(dets,0)):
		bbox = dets[index]
		image = image.copy()
		cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),6)
	return image

def add_identity():
    face_dict = {'Deng Wei':['deng_wei.jpg', 'deng_wei-2.jpg'], 
                 'Shi Yigong':'shi_yigong.jpg',
                 'Xue Qikun':['xue_qikun.jpg', 'xue_qikun-2.jpg'],
                 'Zhang Mu':'zhang_mu.jpg',
                 'Wang Mingzhi':'wang_mingzhi.jpg'}
    for key, value in face_dict.items():
        if not isinstance(value, list):
            value = [value]
        print('add {}:{}'.format(key, value))
        for img_n in value:
            img_path = os.path.join('headteachers', img_n)
            img = cv2.imread(img_path)
            if img is None:
                print('No face found in {}'.format())
                raise FileNotFoundError
            faces = project.get_detected_face(img)
            for idx, face in enumerate(faces):
                cv2.imwrite('{}_face_{}.jpg'.format(img_n[:-4], idx), face)
            project.add_identity_to_database(img, key, detect_before_id=True)

def verifier_test():
    query_faces = []
    img = cv2.imread(os.path.join('headteachers', 'deng_wei-3.jpg'))
    query_faces.append(project.get_detected_face(img)[0])
    img = cv2.imread(os.path.join('headteachers', 'xue_qikun-3.jpg'))
    query_faces.append(project.get_detected_face(img)[0])
    # cv2.imwrite('query_1.jpg', query_faces[0])
    # cv2.imwrite('query_2.jpg', query_faces[1])
    padd_num = 8 - len(query_faces)
    for _ in range(padd_num):
        query_faces.append(np.zeros((96, 112, 3)))
    start = time.time()
    result = project.verification(np.array(query_faces))
    print(result)
    print('Time {}'.format(time.time() - start))

def get_faces():
    img = cv2.imread('sample.jpg')
    faces = project.get_detected_face(img)
    for idx, face in enumerate(faces):
        cv2.imwrite('face_{}.jpg'.format(idx), face)

if __name__ == '__main__':
    verifier_test()
    # add_identity()
    # get_faces()
	
