'''
This is a class to calculate the similarity of two faces 
'''
from __future__ import print_function ,division

import os 
import sys

import numpy as np 
import time 
import os 
FEATURE_LENGTH = 1024;
MAX_MEMORY = 100;
TEST = True;

if TEST:
	caffe_path = '../caffe/python';
	import sys
	sys.path.insert(0,caffe_path);
	import caffe
else:
	import caffe

class Verifier:
	def __init__(self,model_proto, model_weight,ID_List = 'Database/ID_List',Threshold = 0.9):
		self.net = caffe.Net(model_proto, model_weight, caffe.TEST);
		
		self.ID_path = ID_List;
		if not os.path.exists(self.ID_path):
			creat_file = open(self.ID_path,'w');
			creat_file.close();	
		self.ID_List  = self.getidlist(self.ID_path);

		self.Threshold = Threshold;
                # if ID_List length < MAX_MEMORY, we just directly load them into the memory
		self.database = '';
		if len(self.ID_List) < MAX_MEMORY:
			self.database  = np.zeros((len(self.ID_List),FEATURE_LENGTH));
			ID_index = 0;
			for ID in self.ID_List:
				self.database[ID_index,:] = np.load(ID[:-1]).copy();
				ID_index = ID_index + 1;
	
	def getidlist(self,ID_path):
		ID_File = open(ID_path);
		ID_List = ID_File.readlines();
		ID_File.close();
		return ID_List;			 
	def vlen(self,x):
		return np.sqrt(x.dot(x));
        def cosdist(self,o1,o2):
		cosdist = o1.dot(o2) / (self.vlen(o1) * self.vlen(o2) + 1.0e-5);
	def getid(self,feature):
		ID_Name = '';
		max_sim = -1;
		if len(self.database):
			for ID in len(self.database):
				dist = self.cosdist(feature,self.database[ID]);
				if dist> self.Threshold:
					if dist > max_sim:
						ID_Name = self.ID_List[ID][:-1];
		else :
			for ID in self.ID_List:
				data = np.load(ID[:-1]);
				dist = cosdist(feature,data);
				if dist > self.Threshold:
					if dist> max_sim:
						ID_Name = ID[:-1];
		return ID_Name;
	def Verifier(self, face,ID = '' ):
		'''
		Args:
		face: cv2 image 
		ID : Name of this face , for saving to database 
		Return:
		a scalar in (0-1) to measure the similarity between face1 and face2 
		'''
		since = time.time();
		ID_Name = '';
		# resize to 112x96 ,using default mothod 
		face = cv2.resize(face,(112,96));	
		face = (face - 127.5) / 128.0;
		#face = face.transpose((2,0,1));
		net_input = np.zeros((1,face.shape[0],face.shape[1],3));
		net_input[0,:,:,:] = face;
		net_input = net_input.transpose(0,3,1,2);	
		# TODO : here should be take carefully 
		o1 = self.net.forward(data = net_input)['fc5'][0].copy();
		# if ID==null , we search the database to do verification 
		# else we save the ID and corresponding feature vector
		if ID:
			np.save(ID,o1);
			# add ID to ID_List
			ID_File = open(self.ID_path,'a');
			ID_File.write(ID+'\n');
			ID_File.close();
			# return ID_Name	
			return ID_Name;
		else :
			ID_Name = self.getid(o1);
			return ID_Name;

def UNIT():
	caffe.set_mode_gpu();
	caffe.set_device(4);
	model_proto = '../models/verifier/verifier.prototxt';
	model_weight = '../models/verifier/verifier.caffemodel';
	face = np.zeros((112,96,3),dtype = np.float32);
	verify = Verifier(model_proto,model_weight);
	ans = verify.Verifier(face,ID = 'Cheng Jining');
	print(ans);
if __name__ == '__main__':
	UNIT();
