#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import cv2
import numpy as np
import os

class Detector:

    def __init__(self, caffe_model_path):
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
        self.RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
        self.ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
        self.bbox_cache = np.array([]);
        self.image_cache = np.array([]);
        self.point_cache = np.array([]);
    
    def UpdateBBoxCache(self, bboxes, Threshold):
        if self.bbox_cache.shape[0] < 1:
            self.bbox_cache = bboxes;
            return;
        tmp_box = bboxes;
        tmp_index = 0;
        for index in range(bboxes.shape[0]):
            for index2 in range(self.bbox_cache.shape[0]):
                if (self.IOU(bboxes[index,:],self.bbox_cache[index2,:]) > Threshold):
		    tmp_box[tmp_index,:] = self.bbox_cache[index2,:];
                    tmp_index = tmp_index + 1;
                    break; 
                else :
                    tmp_box[tmp_index,:] = bboxes[index,:];
                    tmp_index = tmp_index + 1;
                    break;
        self.bbox_cache = tmp_box[:tmp_index,:];
        #self.bbox_cache = bboxes;
    def UpdateImageCache(self, image):
        self.image_cache = image;
    def UpdatePointCache(self, points, Threshold):
        if self.point_cache.shape[0] < 1:
            self.point_cache = points;
            return;
	tmp_point = points;
        tmp_index = 0;
        for index in range(points.shape[0]):
            for index2 in range(self.point_cache.shape[0]):
                if (self.distance(points[index,:],self.point_cache[index2,:]) < Threshold):
		    tmp_point[tmp_index,:] = self.point_cache[index2,:];
                    tmp_index = tmp_index + 1;
                    break; 
                else :
                    tmp_point[tmp_index,:] = points[index,:];
                    tmp_index = tmp_index + 1;
                    break;
	self.point_cache = tmp_point[:tmp_index,:];
   
    def bbreg(self, boundingbox, reg):
        reg = reg.T
        
        # calibrate bouding boxes
        if reg.shape[1] == 1:
            # print("reshape of reg")
            pass # reshape of reg
        w = boundingbox[:,2] - boundingbox[:,0] + 1
        h = boundingbox[:,3] - boundingbox[:,1] + 1

        bb0 = boundingbox[:,0] + reg[:,0]*w
        bb1 = boundingbox[:,1] + reg[:,1]*h
        bb2 = boundingbox[:,2] + reg[:,2]*w
        bb3 = boundingbox[:,3] + reg[:,3]*h
        
        boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
        #print("bb", boundingbox)
        return boundingbox


    def pad(self, boxesA, w, h):
        boxes = boxesA.copy()
        #print('#################')
        #print('boxes', boxes)
        #print('w,h', w, h)
        
        tmph = boxes[:,3] - boxes[:,1] + 1
        tmpw = boxes[:,2] - boxes[:,0] + 1
        numbox = boxes.shape[0]


        dx = np.ones(numbox)
        dy = np.ones(numbox)
        edx = tmpw 
        edy = tmph

        x = boxes[:,0:1][:,0]
        y = boxes[:,1:2][:,0]
        ex = boxes[:,2:3][:,0]
        ey = boxes[:,3:4][:,0]
       
       
        tmp = np.where(ex > w)[0]
        if tmp.shape[0] != 0:
            edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
            ex[tmp] = w-1

        tmp = np.where(ey > h)[0]
        if tmp.shape[0] != 0:
            edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
            ey[tmp] = h-1

        tmp = np.where(x < 1)[0]
        if tmp.shape[0] != 0:
            dx[tmp] = 2 - x[tmp]
            x[tmp] = np.ones_like(x[tmp])

        tmp = np.where(y < 1)[0]
        if tmp.shape[0] != 0:
            dy[tmp] = 2 - y[tmp]
            y[tmp] = np.ones_like(y[tmp])
        
        # for python index from 0, while matlab from 1
        dy = np.maximum(0, dy-1)
        dx = np.maximum(0, dx-1)
        y = np.maximum(0, y-1)
        x = np.maximum(0, x-1)
        edy = np.maximum(0, edy-1)
        edx = np.maximum(0, edx-1)
        ey = np.maximum(0, ey-1)
        ex = np.maximum(0, ex-1)

        return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



    def rerec(self, bboxA):
        # convert bboxA to square
        w = bboxA[:,2] - bboxA[:,0]
        h = bboxA[:,3] - bboxA[:,1]
        l = np.maximum(w,h).T
        
        bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
        bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
        bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
        return bboxA

    def IOU(self, bboxA, bboxB):
        areaA = (bboxA[2] - bboxA[0]+1) * (bboxA[3] - bboxA[1]+1);
        areaB = (bboxB[2] - bboxB[0]+1) * (bboxB[3] - bboxB[1]+1);
        xx1 = max(bboxA[0],bboxB[0]);
        xx2 = min(bboxA[2],bboxB[2]);
        yy1 = max(bboxA[1],bboxB[1]);
        yy2 = min(bboxA[3],bboxB[3]);
        w = np.maximum(0.0, xx2 - xx1 + 1);
        h = np.maximum(0.0, yy2 - yy1 + 1);
        inter = w * h
        iou = inter / (areaA + areaB - inter);
        
        return iou;

    def distance(self, pointA, pointB):
        sum = 0
        for i in range(1,10):
	    sum = sum + abs(pointA[i] - pointB[i])
	return sum

    def nms(self, boxes, threshold, type):
        """nms
        :boxes: [:,0:5]
        :threshold: 0.5 like
        :type: 'Min' or others
        :returns: TODO
        """
        if boxes.shape[0] == 0:
            return np.array([])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort()) # read s using I
        
        pick = [];
        while len(I) > 0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type == 'Min':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where( o <= threshold)[0]]
        return pick


    def generateBoundingBox(self, map, reg, scale, t):
        stride = 2
        cellsize = 12
        map = map.T
        dx1 = reg[0,:,:].T
        dy1 = reg[1,:,:].T
        dx2 = reg[2,:,:].T
        dy2 = reg[3,:,:].T
        (x, y) = np.where(map >= t)

        yy = y
        xx = x
        
        score = map[x,y]
        reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

        if reg.shape[0] == 0:
            pass
        boundingbox = np.array([yy, xx]).T

        bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
        bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
        score = np.array([score])

        boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

        return boundingbox_out.T



    def drawBoxes(self, im, boxes,points):
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        '''
        for i in range(x1.shape[0]):
            cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 6);
            for index in range(0,5):

            	cv2.circle(im,(int(points[i][index]),int(points[i][index+5])),3,(0,255,0),-1);
        '''
        x3 = x1 + 0.25 * (x2 - x1)
        x4 = x2 - 0.25 * (x2 - x1)
        y3 = y1 + 0.25 * (y2 - y1)
        y4 = y2 - 0.25 * (y2 - y1)
        for i in range(x1.shape[0]):
            cv2.line(im, (int(x1[i]), int(y1[i])), (int(x3[i]), int(y1[i])), (0,255,0), 2);
            cv2.line(im, (int(x1[i]), int(y1[i])), (int(x1[i]), int(y3[i])), (0,255,0), 2);
	    cv2.line(im, (int(x2[i]), int(y1[i])), (int(x4[i]), int(y1[i])), (0,255,0), 2);
	    cv2.line(im, (int(x2[i]), int(y1[i])), (int(x2[i]), int(y3[i])), (0,255,0), 2);
	    cv2.line(im, (int(x1[i]), int(y2[i])), (int(x3[i]), int(y2[i])), (0,255,0), 2);
	    cv2.line(im, (int(x1[i]), int(y2[i])), (int(x1[i]), int(y4[i])), (0,255,0), 2);
	    cv2.line(im, (int(x2[i]), int(y2[i])), (int(x4[i]), int(y2[i])), (0,255,0), 2);
	    cv2.line(im, (int(x2[i]), int(y2[i])), (int(x2[i]), int(y4[i])), (0,255,0), 2);
	    for index in range(0,5):
            	cv2.circle(im,(int(points[i][index]),int(points[i][index+5])),1,(0,255,0),-1);

        return im

    from time import time
    _tstart_stack = []
    def tic():
        _tstart_stack.append(time())
    def toc(fmt="Elapsed: %s s"):
        print(fmt % (time()-_tstart_stack.pop()))

    def detect_face(self, img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):       
        
        img2 = img.copy()

        factor_count = 0
        total_boxes = np.zeros((0,9), np.float)
        points = []
        verify_features = []
        h = img.shape[0]
        w = img.shape[1]
        minl = min(h, w)
        img = img.astype(float)
        m = 12.0/minsize
        minl = minl*m
        
        import time 
        start = time.time();
        # create scale pyramid
        scales = []
        while minl >= 12:
            scales.append(m * pow(factor, factor_count))
            minl *= factor
            factor_count += 1
        # first stage
        for scale in scales:
            hs = int(np.ceil(h*scale))
            ws = int(np.ceil(w*scale))

            if fastresize:
                im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
                im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
            else: 
                im_data = cv2.resize(img, (ws,hs)) # default is bilinear
                im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]

            im_data = np.swapaxes(im_data, 0, 2)
            im_data = np.array([im_data], dtype = np.float)
            PNet.blobs['data'].reshape(1, 3, ws, hs)
            PNet.blobs['data'].data[...] = im_data
            out = PNet.forward()
        
            boxes = self.generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
            if boxes.shape[0] != 0:
                pick = self.nms(boxes, 0.5, 'Union')

                if len(pick) > 0 :
                    boxes = boxes[pick, :]

            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)
	    end = time.time();


        #####
        # 1 #
        #####
        numbox = total_boxes.shape[0]
        if numbox > 0:
            # nms
            pick = self.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            #print("[2]:",total_boxes.shape[0])
            
            # revise and convert to square
            regh = total_boxes[:,3] - total_boxes[:,1]
            regw = total_boxes[:,2] - total_boxes[:,0]
            t1 = total_boxes[:,0] + total_boxes[:,5]*regw
            t2 = total_boxes[:,1] + total_boxes[:,6]*regh
            t3 = total_boxes[:,2] + total_boxes[:,7]*regw
            t4 = total_boxes[:,3] + total_boxes[:,8]*regh
            t5 = total_boxes[:,4]
            total_boxes = np.array([t1,t2,t3,t4,t5]).T


            total_boxes = self.rerec(total_boxes) # convert box to square
            #print("[4]:",total_boxes.shape[0])
            
            total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])

            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)



        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage

            # construct input for RNet
            tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
              
                

                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                
                tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
                
            tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

            

            # RNet

            tempimg = np.swapaxes(tempimg, 1, 3)
            #print(tempimg[0,:,0,0])
            
            RNet.blobs['data'].reshape(numbox, 3, 24, 24)
            RNet.blobs['data'].data[...] = tempimg
            out = RNet.forward()


            score = out['prob1'][:,1]
            pass_t = np.where(score>threshold[1])[0]
            
            score =  np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
            
            mv = out['conv5-2'][pass_t, :].T
            if total_boxes.shape[0] > 0:
                pick = self.nms(total_boxes, 0.7, 'Union')
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    total_boxes = self.bbreg(total_boxes, mv[:, pick])
                    total_boxes = self.rerec(total_boxes)


            numbox = total_boxes.shape[0]
            if numbox > 0:
                # third stage
                
                total_boxes = np.fix(total_boxes)
                [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)
               
                tempimg = np.zeros((numbox, 48, 48, 3))
                for k in range(numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                    tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                    tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
                tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                    
                # ONet
                tempimg = np.swapaxes(tempimg, 1, 3)
                ONet.blobs['data'].reshape(numbox, 3, 48, 48)
                ONet.blobs['data'].data[...] = tempimg
                out = ONet.forward()
                                
                score = out['prob1'][:,1]
                points = out['conv6-3']
                
                verify_features = ONet.blobs['conv5'].data
                
                pass_t = np.where(score>threshold[2])[0]
                points = points[pass_t, :]
                
                verify_features = verify_features[pass_t, :]
                
                
                score = np.array([score[pass_t]]).T
                total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
                
                mv = out['conv6-2'][pass_t, :].T
                w = total_boxes[:,3] - total_boxes[:,1] + 1
                h = total_boxes[:,2] - total_boxes[:,0] + 1

                points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
                points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

                if total_boxes.shape[0] > 0:
                    total_boxes = self.bbreg(total_boxes, mv[:,:])
                    pick = self.nms(total_boxes, 0.7, 'Min')
                    
                    
                    if len(pick) > 0 :
                        total_boxes = total_boxes[pick, :]
                        points = points[pick, :]
                        verify_features = verify_features[pick, :]
                        
        

        return total_boxes, points, verify_features


    def haveFace(self, img, facedetector):
        minsize = facedetector[0]
        PNet = facedetector[1]
        RNet = facedetector[2]
        ONet = facedetector[3]
        threshold = facedetector[4]
        factor = facedetector[5]
        
        if max(img.shape[0], img.shape[1]) < minsize:
            return False, []

        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        
        #tic()
        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
        #toc()
        containFace = (True, False)[boundingboxes.shape[0]==0]
        return containFace, boundingboxes

    def resize_img(self, img, frame, bboxes, points):
        
        points = np.array(points);
        original_h  = img.shape[0];
        original_w = img.shape[1];
        target_h = frame.shape[0];
        target_w = frame.shape[1];
        
        ratio = np.zeros(bboxes.shape);
        new_points = np.zeros(points.shape);
        
        ratio[:,0] = bboxes[:,0] * target_w / original_w;
        ratio[:,1] = bboxes[:,1] * target_h/ original_h;
        ratio[:,2] = bboxes[:,2] * target_w/ original_w;
        ratio[:,3] = bboxes[:,3] * target_h/ original_h;
        
        new_points[:,:5] = points[:,:5] * target_w/ original_w;
        new_points[:,5:] = points[:,5:] * target_h / original_h;
        return ratio, new_points;
        
    def detect(self, frame, w, h, minsize, threshold, factor):
        
        img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC);
        
        PNet = self.PNet;
        RNet = self.RNet;
        ONet = self.ONet;
        
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        
        boundingboxes, points, verify_features = self.detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor);

        if len(boundingboxes) < 1:
            return  boundingboxes, points, verify_features;
        boundingboxes ,points= self.resize_img(img, frame, boundingboxes, points);
        return boundingboxes, points, verify_features
