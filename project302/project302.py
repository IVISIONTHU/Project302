import sys
import detector
import verifier
import cv2
import numpy as np
import config as cfg

class Project302:
	def __init__(self, detect_interval, max_face, show_result=False, do_verification=False):
		print('init Project302\n');
		self.detect_interval = detect_interval;
		self.max_face = max_face;
		self.frame = 0;
		self.detector = None;
		self.verifier = None;
		self.show_result = show_result;
		self.do_verification = do_verification
		# load detection & tracking parameters
		self.detect_w = cfg.detect_w;
		self.detect_h = cfg.detect_h;
		self.detect_minsize = cfg.detect_minsize;
		self.detect_threshold = cfg.detect_threshold;
		self.detect_factor = cfg.detect_factor;
		self.track_minsize = cfg.detect_minsize;
		self.track_threshold = cfg.track_threshold;
		self.track_factor = cfg.track_factor;		
                self.track_crop = cfg.track_crop;
                self.boxnum = cfg.track_boxnum;
                self.point_thresh = cfg.point_thresh

	def init_detector(self, caffe_model_path):
		self.detector = detector.Detector(caffe_model_path);
		print('detector init success');

	def init_verifier(self,model_proto,model_weight):
	    self.verifier = verifier.Verifier(model_proto, 
                                              model_weight, 
                                              database_root='../verifier/Database',
                                              Threshold=0.2)
	    print('verifier init success')

        def add_identity_to_database(self, img, ID, detect_before_id=False):
            if len(np.array(img).shape) == 3:
                if detect_before_id:
                    img, points = self.get_detected_face(img, single_face=True)
                    if None in img:
                        return False
                status = self.verifier.Verifier([img], points, save_id=True, ID=ID)
            else:
                if detect_before_id:
                    face_list = [None for _ in range(len(img))]
                    point_list = [None for _ in range(len(img))]
                    for idx, _img in enumerate(img):
                        face_list[idx], point_list[idx] = self.get_detected_face(_img, single_face=True)
                        if face_list[idx] is None:
                            return False

                for i in range(0, len(img), cfg.max_face):
                    t1 = i
                    t2 = min(i+cfg.max_face, len(img))
                    status = self.verifier.Verifier(img[t1:t2], point_list[t1:t2], save_id=True, ID=ID[t1:t2])
                    if not status:
                        print('{} Identity addition failed'.format(ID[t1:t2]))
            return True

        def verification(self, img, keypoint):
            if len(np.array(img).shape) == 3:
                img = img[np.newaxis, :]
            ids = self.verifier.Verifier(img, keypoint)
            return ids

        def get_detected_face(self, img, single_face=False):
            bboxes, points, _ = self.detector.detect(img, 
                                self.detect_w, 
                                self.detect_h, 
                                self.detect_minsize, 
                                self.detect_threshold, 
                                self.detect_factor)

            points = np.array([[[points[j][i] - bboxes[j][0], points[j][i+5] - [bboxes[j][1]]] 
                                            for i in range(5)] for j in range(len(points))])
            if len(bboxes) < 1:
                print('No face found in image')
                # raise TypeError
                return None, None
            if single_face:
                #if len(bboxes) > 1:
                    #print('Containing multiple faces')
                    # raise TypeError
                bboxes = bboxes[0].astype(np.int)
                img = img[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2], :]
                return img, np.array(points[0])
            else:
                imgs = [img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] for bbox in bboxes.astype(np.int)]
                return imgs, points

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

        def UpdateCache(self, bboxes, points, ids=None, bbox_threshold=0.8, point_threshold=10.0):
            if self.detector.bbox_cache.shape[0] < 1:
                self.detector.bbox_cache = bboxes
                self.detector.point_cache = points
                if ids is not None:
                    self.verifier.id_cache = ids
                return
            tmp_box = bboxes;
            if ids is not None:
                tmp_id = ids
            tmp_index = 0;
            for index in range(bboxes.shape[0]):
                FOUND = False
                for index2 in range(self.detector.bbox_cache.shape[0]):
                    if (self.detector.IOU(bboxes[index,:], self.detector.bbox_cache[index2,:]) > bbox_threshold):
		        tmp_box[tmp_index,:] = self.detector.bbox_cache[index2, :]
                        if ids is not None:
                            tmp_id[tmp_index] = self.verifier.id_cache[index2]
                        tmp_index = tmp_index + 1;
                        FOUND = True
                        break; 
                if not FOUND:
                    tmp_box[tmp_index,:] = bboxes[index,:]
                    if ids is not None:
                        tmp_id[tmp_index] = ids[index]
                    tmp_index = tmp_index + 1;

            self.detector.bbox_cache = tmp_box[:tmp_index, :];
            if ids is not None:
                self.verifier.id_cache = tmp_id[:tmp_index]

            tmp_index = 0;
            tmp_point = points
            for index in range(bboxes.shape[0]):
                FOUND = False
                for index2 in range(self.detector.point_cache.shape[0]):
                    if (self.detector.distance(points[index,:],self.detector.point_cache[index2,:]) < point_threshold):
		        tmp_point[tmp_index,:] = self.detector.point_cache[index2,:];
                        tmp_index = tmp_index + 1;
                        FOUND = True
                        break; 
                if not FOUND:
                    tmp_point[tmp_index,:] = points[index,:];
                    tmp_index = tmp_index + 1;
            self.detector.point_cache = tmp_point[:tmp_index,:];

        def draw(self, image, bboxes, points, ids=None):
	    image = self.detector.drawBoxes(image, bboxes, points)
            if ids is None:
                return image
            for _id, _bbox in zip(ids, bboxes):
                if _id is None:
                    continue
                cv2.putText(image, _id, 
                    (int((_bbox[0] + _bbox[3])/2), int(_bbox[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
            return image
            

	def Surveillance(self, image):
		'''
		This is an interface for surveillance project
		
		it receives an image as an input and output an image with 
		
		bounding boxes & landmarks drawn 
		
		'''
	        self.frame = self.frame + 1;
                bboxes = [];
                points = [];
                verify_features = [];
                face_index = 0;
		image_result = image.copy();
		if(self.frame % self.detect_interval == 0):
			# detect		
			bboxes, points, verify_features = self.detector.detect(image, self.detect_w, self.detect_h, self.detect_minsize, self.detect_threshold, self.detect_factor, 9999);
			face_index = bboxes.shape[0];
                        #self.detector.UpdateBBoxCache(bboxes); 
			# if (len(bboxes) >= 1):
			# 	self.detector.UpdateBBoxCache(bboxes[:min(face_index,self.max_face),:], 0.8); 
			# 	self.detector.UpdatePointCache(points[:min(face_index,self.max_face),:], 5); 
			# 	image_result = self.detector.drawBoxes(image, self.detector.bbox_cache, points);
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
					crop_factor = (self.track_crop - 1) / 2
					# crop_factor = 0.25;

                                        crop_w = self.detector.bbox_cache[k, 2] - self.detector.bbox_cache[k, 0];
					crop_h = self.detector.bbox_cache[k, 3] - self.detector.bbox_cache[k, 1];
				        	
					crop_x1 = int(self.detector.bbox_cache[k, 0] - crop_w * crop_factor);
					crop_y1 = int(self.detector.bbox_cache[k, 1] - crop_h * crop_factor);
					crop_x2 = int(self.detector.bbox_cache[k, 2] + crop_w * crop_factor);
					crop_y2 = int(self.detector.bbox_cache[k, 3] + crop_h * crop_factor);
					tmp_image = np.zeros((crop_y2 - crop_y1 + 1, crop_x2 - crop_x1 + 1, 3))
					crop_image= image[max(0, crop_y1):min(image.shape[0], crop_y2) + 1, max(0, crop_x1):min(image.shape[1], crop_x2) + 1]
					padx = tmp_image.shape[1] - crop_image.shape[1]
					pady = tmp_image.shape[0] - crop_image.shape[0]
					tmp_image = cv2.copyMakeBorder(crop_image, 0, pady, 0, padx, cv2.BORDER_CONSTANT)
					# cv2.imshow('test', tmp_image)
					crop_x1 = max(0, crop_x1)
					crop_y1 = max(0, crop_y1)

                                        '''
					tmpx1 = max(0, 0 - crop_x1)
					tmpx2 = min(crop_x2 - crop_x1, image.shape[1] - crop_x1)
					tmpy1 = max(0, 0 - crop_y1)
					tmpy2 = min(crop_y2 - crop_y1, image.shape[0] - crop_y1)
					tmp_image[tmpy1:tmpy2, tmpx1:tmpx2] = image[max(0, crop_y1):min(image.shape[0], crop_y2), max(0, crop_x1):min(image.shape[1], crop_x2)]
                                        '''
					#  by dalong : w = h = 48;
					w = 48 ;
                                        h = 48 * tmp_image.shape[0]/ tmp_image.shape[1];
					#cv2.imshow('test'+str(k), tmp_image)
					#if (bboxes.shape[0] == 0):
					tmp_bboxes, tmp_points, tmp_verify_features = self.detector.detect(tmp_image, w, h, self.track_minsize, self.track_threshold, self.track_factor, self.boxnum);
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
                                '''
				if (len(bboxes) >= 1):
					self.detector.UpdateBBoxCache(bboxes[:min(face_index,self.max_face),:], 0.8); 
					self.detector.UpdatePointCache(points[:min(face_index,self.max_face),:], 10); 
					image_result = self.detector.drawBoxes(image, self.detector.bbox_cache, points);
                                '''
							
		# verify
		'''
		verify_features is an 256*n array where n represents the number of bounding boxes
		'''
		if (len(bboxes) >= 1):
                    _bbox = bboxes[:min(face_index, self.max_face),:]
                    _point = points[:min(face_index, self.max_face),:]
                    if self.do_verification and self.frame % self.detect_interval == 0:
                        _bbox = np.maximum(_bbox, 0.)
                        _point = np.maximum(_point, 0.)
                        faces = [image[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in _bbox.astype(np.int)]
                        _points = np.array([[[points[i][j] - _bbox[i][0], points[i][j+5] - _bbox[i][1]] 
                                         for j in range(5)] for i in range(len(_bbox))])
                        ids = self.verification(faces, _points)
                        # _id = ids[:min(face_index, self.max_face),:]
		        self.UpdateCache(_bbox, _point, np.array(ids), 0.8, 10)
                        image_result = self.draw(image, self.detector.bbox_cache, self.detector.point_cache, self.verifier.id_cache)
                    else:
		        self.UpdateCache(_bbox, _point, bbox_threshold=0.8, point_threshold=self.point_thresh)
                        image_result = self.draw(image, self.detector.bbox_cache, self.detector.point_cache)

		    # self.detector.UpdatePointCache(points[:min(face_index,self.max_face),:], 10); 
		    # image_result = self.detector.drawBoxes(image, self.detector.bbox_cache, points)
		    # image_result = self.detector.drawBoxes(image, self.detector.bbox_cache, self.detector.point_cache)

		return image_result
