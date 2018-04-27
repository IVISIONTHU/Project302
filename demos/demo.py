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

project = Project302(detect_interval=cfg.detect_interval, 
                     max_face=cfg.max_face,
                     do_verification=True)

project.init_detector(cfg.detector_mtcnn);
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
	caffe.set_mode_gpu();
	caffe.set_device(cfg.GPU_DEVICE);
else:
	caffe.set_mode_cpu();

def show_result(image,dets):
	#image = image[:,:,(2,1,0)];
	if(np.size(dets) ==0):
		return image
	for index in xrange(np.size(dets,0)):
		bbox = dets[index];
		image = image.copy()
		cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),6);
	return image;
'''
def show_result(image,bboxes):
		import matplotlib.pyplot as plt;
		image = image[:,:,(2,1,0)];
		for index in xrange(bboxes.shape[0]):
			bbox = bboxes[index];
			plt.cla();
			plt.imshow(image);
			plt.gca().add_patch(
				plt.Rectangle((bbox[0],bbox[1]),
					       bbox[2] - bbox[0],
					       bbox[3] - bbox[1],
					       fill =  False,
					       edgecolor = 'g',
					       linewidth = 3)
			);
			plt.show();
'''
def demo():
	print('\n\n\n\n\n\n\n\n\n');
	cap = cv2.VideoCapture(cfg.CAMERA_INDEX);
	cap.set(3, 1920);
	cap.set(4, 1080);
	ret, frame = cap.read();
        frame_index = cfg.frame_skip;
	window_name = 'test_win'
	cv2.namedWindow(window_name)
	while(ret):
		start = time.time();
		ret,frame = cap.read();
		#frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                if not ret:
                    continue;
		if frame_index != cfg.frame_skip:
                    frame_index = frame_index + 1;
                    continue;
		#cv2.imshow('test',frame);
		image = project.Surveillance(frame);
		frame_index = 0;
		end = time.time();
		print('log project time = {}\n'.format(end - start));
		#image = image[:,:,::-1]
		#image = cv2.resize(image, (1080, 960),interpolation=cv2.INTER_CUBIC)
		#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
		# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow(window_name, image);
		cv2.waitKey(33);

if __name__ == '__main__':
	demo();
	
