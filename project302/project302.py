import sys
import detector
#import verifier
import cv2
import numpy as np
import config as cfg

class Project302:
	def __init__(self,detect_interval,max_face,show_result = False,do_verfiy = False):
		print('init Project302\n');
		self.detect_interval = detect_interval;
		self.max_face = max_face;
		self.frame = 0;
		self.detector = None;
		#self.verifier = None;
		self.show_result =show_result;
		#self.do_verify = do_verify;
		# load detection & tracking parameters
		self.detect_w = cfg.detect_w;
		self.detect_h = cfg.detect_h;
		self.detect_minsize = cfg.detect_minsize;
		self.detect_threshold = cfg.detect_threshold;
		self.detect_factor = cfg.detect_factor;
		self.track_minsize = cfg.detect_minsize;
		self.track_threshold = cfg.track_threshold;
		self.track_factor = cfg.track_factor;		


	def init_detector(self, caffe_model_path):
		self.detector = detector.Detector(caffe_model_path);
		print('detector init success');


	def init_verifier(self,model_proto,model_weight):
		self.verifier = verifier.Verifier(model_proto,model_weight);
		print('verifier init success');	

	def Surveillance(self, image):
		'''
		This is an interface for surveillance project
		
		it receives an image as an input and output an image with 
		
		bounding boxes & landmarks drawn 
		
		'''
		self.frame = self.frame + 1;
		if(self.frame % self.detect_interval == 0):
			# detect		
			image_result, bboxes, points, verify_features = self.detector.detect(image, self.detect_w, self.detect_h, self.detect_minsize, self.detect_threshold, self.detect_factor);
			self.detector.UpdateImageCache(image);
			self.detector.UpdateBBoxCache(bboxes); 
			self.detector.UpdatePointCache(points);
		else:
			# track
			pass;
			
		# verify
		'''
		verify_features is an 256*n array where n represents the number of bounding boxes
		
		'''
		
		return image_result;
	
