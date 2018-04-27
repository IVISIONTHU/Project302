#!/usr/bin/env python
import cv2, os, sys
import init_path
from project302 import Project302
from alignment import alignment
import matplotlib.pyplot as plt 
import config as cfg
import caffe 
import numpy as np
import time 
import json

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

project = Project302(detect_interval=1, verify_interval=20, max_face=8, do_verification=True)

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
    '''
    face_dict = {'Deng Wei':'deng_wei-2.jpg', 
                 'Shi Yigong':'shi_yigong.jpg',
                 'Xue Qikun':['xue_qikun.jpg', 'xue_qikun-2.jpg'],
                 'Zhang Mu':'zhang_mu.jpg',
                 'Wang Mingzhi':'wang_mingzhi.jpg',
                 'Da long':'dalong_0.jpg',
                 'Wang Qian':'wangq_0.jpg',
                 'Chen Jining':'chen_jining.jpg',
                 'Chen Xi':'chen_xi.jpg',
                 'Gu Binglin':'gu_binglin.jpg',
                 'He Meiying':'he_meiying.jpg',
                 'Wang Dazhong':['wang_dazhong.jpg', 'wang_dazhong-2.jpg'],
                 'Wang Mingzhi':['wang_mingzhi.jpg'],
                 'Zhang Xiaowen':'zhang_xiaowen.jpg'}
    '''
    with open(os.path.join('headteachers', 'name_list.json'), 'r') as f:
        # json.dump(face_dict, f)
        face_dict = json.load(f)
    for key, value in face_dict.items():
        key_list = []
        value_list = []
        if not isinstance(value, list):
            key_list.append(key)
            value_list.append(value)
        else:
            for i in range(len(value)):
                key_list.append(key)
                value_list.append(value[i])
        # print('add {}:{}'.format(key, value))
        img_list = [cv2.imread(os.path.join('headteachers', img_n)) for img_n in value_list]
        # face_list = [None for _ in range(len(img_list))]
        # point_list = [None for _ in range(len(img_list))]
        # for idx, img in enumerate(img_list):
        #     face_list[idx], point_list[idx] = project.get_detected_face(img, single_face=True)
        project.add_identity_to_database(img_list, key_list, detect_before_id=True)

        '''
        for img_n in value:
            img_path = os.path.join('headteachers', img_n)
            img = cv2.imread(img_path)
            if img is None:
                print('No face found in {}'.format())
                continue
                # raise FileNotFoundError
            faces, keypoints = project.get_detected_face(img)
            for idx, face in enumerate(faces):
                cv2.imwrite('{}_face_{}.jpg'.format(img_n[:-4], idx), face)
                aligned_face = alignment(face, keypoints[idx])
                cv2.imwrite('{}_aligned_face_{}.jpg'.format(img_n[:-4], idx), aligned_face)
            project.add_identity_to_database(img, key, detect_before_id=True)
        '''

def verifier_test():
    query_faces = []
    query_keypoints = []

    img = cv2.imread(os.path.join('headteachers', 'deng_wei-3.jpg'))
    face, keypoint = project.get_detected_face(img)
    query_faces.append(face[0])
    query_keypoints.append(keypoint[0])

    img = cv2.imread(os.path.join('headteachers', 'you_zheng.jpg'))
    face, keypoint = project.get_detected_face(img)
    query_faces.append(face[0])
    query_keypoints.append(keypoint[0])
    padd_num = 8 - len(query_faces)
    for _ in range(padd_num):
        query_faces.append(np.random.random((96, 112, 3)))
        query_keypoints.append(np.random.random((5, 2)))
    start = time.time()
    result = project.verification(query_faces, query_keypoints)
    print(result)
    print('Time {}'.format(time.time() - start))

def get_faces():
    img = cv2.imread('sample.jpg')
    faces = project.get_detected_face(img)
    for idx, face in enumerate(faces):
        cv2.imwrite('face_{}.jpg'.format(idx), face)

def online_test():
    frame_index = 1;
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX);
    cap.set(3, 1920);
    cap.set(4, 1080);
    ret ,frame = cap.read();
    frame_index = 1;
    while(ret):
	    start = time.time();
	    ret,frame = cap.read();	
	    #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            if not ret:
                continue;
            if frame_index % 3    !=0:
                frame_index = frame_index +1 ;
                continue;
	    #cv2.imshow('test',frame);
	    # image = project.Surveillance(frame);
            face, keypoint = project.get_detected_face(frame, single_face=True)
            if face is None:
                continue
            IDs = ['' for _ in range(len(face))]
            for i in range(0, len(face), cfg.verification_batch_size):
                t1 = i
                t2 = min(i+cfg.verification_batch_size, len(face))
                IDs[t1:t2] = project.verification(np.array(face[t1:t2]), keypoint[t1:t2])

            # result = project.verification(np.array(face), keypoint)
	    frame_index = frame_index + 1;
	    end = time.time();
	    print('log project time {} | ID {}\n'.format(end - start, IDs));
	    #image = image[:,:,::-1]
	    #image = cv2.resize(image, (1080, 960),interpolation=cv2.INTER_CUBIC)
	    window_name = 'test_win'
	    # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
	    cv2.namedWindow(window_name)
	    # cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	    cv2.imshow(window_name, frame);
	    cv2.waitKey(33);


if __name__ == '__main__':
    # verifier_test()
    # online_test()
    add_identity()
    # get_faces()
	
