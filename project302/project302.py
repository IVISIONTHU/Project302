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
        def Filter(self,bbox,points,features):
            if bbox.shape[0] < 1:
                return None,None,None;
            conf = -1;
            index = 0;
            for i in range(bbox.shape[0]):
                if bbox[i][4] > conf:
                    conf = bbox[i][4];
                    index = i;
            return bbox[index,:],points[index,:],features[index,:];
	def Surveillance(self, image):
		'''
		This is an interface for surveillance project
		
		it receives an image as an input and output an image with 
		
		bounding boxes & landmarks drawn 
		
		'''
	        self.frame = self.frame + 1;
                bboxes =  [];
                points = [];
                verify_features = [];
                face_index = 0;
		image_result = image.copy();
		if(self.frame % self.detect_interval == 0):
			# detect		
			bboxes, points, verify_features = self.detector.detect(image, self.detect_w, self.detect_h, self.detect_minsize, self.detect_threshold, self.detect_factor);
			face_index = bboxes.shape[0];
                        #self.detector.UpdateBBoxCache(bboxes); 
			if (len(bboxes) >= 1):
				self.detector.UpdateBBoxCache(bboxes[:min(face_index,self.max_face),:], 0.7); 
				self.detector.UpdatePointCache(points[:min(face_index,self.max_face),:], 30); 
				image_result = self.detector.drawBoxes(image, self.detector.bbox_cache, points);
			#if (len(bboxes) < 1):
			#    image_result = image.copy();
			#else:
			#	image_result = self.detector.drawBoxes(image, bboxes, points);
		else:
			# track
			if (len(self.detector.bbox_cache) >= 1):
				numbox = self.detector.bbox_cache.shape[0]
				bboxes = np.zeros((numbox, 5))
				points = np.zeros((numbox, 10))
				verify_features = np.zeros((numbox, 256), np.float);
				# by dalong : load all faces as a batch
				# crop -> resize (48 48)- > batch -> net  
				for k in range(numbox):
					crop_factor = 0.75
					# crop_factor = 0.25;

                                        crop_w = self.detector.bbox_cache[k, 2] - self.detector.bbox_cache[k, 0];
					crop_h = self.detector.bbox_cache[k, 3] - self.detector.bbox_cache[k, 1];
				        	
					crop_x1 = int(self.detector.bbox_cache[k, 0] - crop_w * crop_factor);
					crop_y1 = int(self.detector.bbox_cache[k, 1] - crop_h * crop_factor);
					crop_x2 = int(self.detector.bbox_cache[k, 2] + crop_w * crop_factor);
					crop_y2 = int(self.detector.bbox_cache[k, 3] + crop_h * crop_factor);
					tmp_image = np.zeros((crop_y2 - crop_y1 + 1, crop_x2 - crop_x1 + 1, 3))
					tmpx1 = max(0, 0 - crop_x1)
					tmpx2 = min(crop_x2 - crop_x1, image.shape[1] - crop_x1)
					tmpy1 = max(0, 0 - crop_y1)
					tmpy2 = min(crop_y2 - crop_y1, image.shape[0] - crop_y1)
					tmp_image[tmpy1:tmpy2, tmpx1:tmpx2] = image[max(0, crop_y1):min(image.shape[0], crop_y2), max(0, crop_x1):min(image.shape[1], crop_x2)]
					#  by dalong : w = h = 48;
					w = 48 ;
                                        h = 48 * tmp_image.shape[0]/ tmp_image.shape[1];
					#cv2.imshow('test'+str(k), tmp_image)
					#if (bboxes.shape[0] == 0):
					tmp_bboxes, tmp_points, tmp_verify_features = self.detector.detect(tmp_image, w, h, self.track_minsize, self.track_threshold, self.track_factor);
					points = np.array(points);

                                        tmp_bboxes , tmp_points, tmp_verify_features = self.Filter(tmp_bboxes,tmp_points,tmp_verify_features);
                                        if tmp_bboxes is None :
                                            continue ;
					bboxes[face_index, 0] = tmp_bboxes[0] + crop_x1;
					bboxes[face_index, 1] = tmp_bboxes[1] + crop_y1;
					bboxes[face_index, 2] = tmp_bboxes[2] + crop_x1;
					bboxes[face_index, 3] = tmp_bboxes[3] + crop_y1;
					points[face_index, 0:5] = tmp_points[0:5] + crop_x1;
					points[face_index, 5:10] = tmp_points[5:10] + crop_y1;
					verify_features[face_index,:] = tmp_verify_features;
                                        face_index = face_index + 1;				
				if (len(bboxes) >= 1):
					self.detector.UpdateBBoxCache(bboxes[:min(face_index,self.max_face),:], 0.7); 
					self.detector.UpdatePointCache(points[:min(face_index,self.max_face),:], 80); 
					image_result = self.detector.drawBoxes(image, self.detector.bbox_cache, points);
							
		# verify
		'''
		verify_features is an 256*n array where n represents the number of bounding boxes
		
		'''
		
		return image_result;
	
