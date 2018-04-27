'''
This is a class to calculate the similarity of two faces 
'''
from __future__ import print_function, division
import numpy as np
import os, sys, time
import cv2
from alignment import alignment
import config as cfg

FEATURE_LENGTH = 512
MAX_MEMORY = 1000
TEST = True

if TEST:
    caffe_path = '/usr/local/caffe/python'
    import sys
    sys.path.insert(0, caffe_path)
    import caffe
else:
    import caffe

class Verifier:
    def __init__(self, model_proto, model_weight, database_root='Database', Threshold=0.9):
        self.net = caffe.Net(model_proto, model_weight, caffe.TEST)
        self.database_root = database_root
	self.ID_path = os.path.join(database_root, 'ID_List')
        self.id_cache = np.array([])
        # [id, ftrs]
	if not os.path.exists(self.ID_path):
	    open(self.ID_path, 'w').close()
	    # creat_file.close()
	self.ID_List  = self.get_id_list(self.ID_path)
	self.Threshold = Threshold
        # if ID_List length < MAX_MEMORY, we just directly load them into the memory
	if len(self.ID_List) < MAX_MEMORY:
            self.database = [[] for _ in range(len(self.ID_List))]
	    ID_index = 0
	    for idx, ID in enumerate(self.ID_List):
                self.database[idx] = np.load(os.path.join(self.database_root, ID[:-1]+'.npy'))
        else:
            self.database = None

    def get_id_list(self, ID_path):
        with open(self.ID_path, 'r') as f:
            ID_List = f.readlines()
	return ID_List

    def vlen(self, x):
	return np.sqrt(x.dot(x))

    def cosdist(self, o1, o2):
	cosdist = o1.dot(o2) / (self.vlen(o1) * self.vlen(o2) + 1.0e-5)

    def dist(self, ftr1, ftr2, dist_type='euclidean'):
        diff = np.expand_dims(ftr1, 1) - np.expand_dims(ftr2, 0)
        if dist_type == 'cityblock':
            return np.sum(np.abs(diff), 2)
        elif dist_type == 'sqeuclidean':
            return np.sqrt(np.sum(diff ** 2, 2))
        elif dist_type == 'cosine':
            ftr1 = np.expand_dims(ftr1 / np.sqrt(np.sum(ftr1 ** 2, 1, keepdims=True)), 1)
            ftr2 = np.expand_dims(ftr2 / np.sqrt(np.sum(ftr2 ** 2, 1, keepdims=True)), 0)
            return np.sum(ftr1 * ftr2, 2)
        elif dist_type == 'euclidean_norm':
            ftr1 = ftr1 / np.sqrt(np.sum(ftr1 ** 2))
            ftr2 = ftr2 / np.sqrt(np.sum(ftr2 ** 2))
            return np.sum(diff ** 2, 2)
        else:
            return np.sum(diff ** 2, 2)

    def getid(self, feature):
        # candidates [id, similarity]
        candidates = [[] for _ in range(len(feature))]

	if self.database is not None:
	    for ID in range(len(self.database)):
		# dist = self.cosdist(feature,self.database[ID])
		dists = self.dist(feature, self.database[ID], 'cosine');
                print('Dist for {} | {}'.format(self.ID_List[ID][:-1], dists))

                for idx, distance in enumerate(dists):
                    # print('distance {}'.format(distance))
                    max_idx = np.argmax(distance)
                    if distance[max_idx] > self.Threshold:
                        candidates[idx].append([self.ID_List[ID][:-1], distance[max_idx]])
	else:
	    for ID in self.ID_List:
                ID_bucket = os.path.join(self.database_root, ID[:-1]+'.npy')
		data = np.load(ID_bucket)
		# dist = cosdist(feature, data)
		dists = self.dist(feature, data, 'cosine')
                for idx, distance in enumerate(dists):
                    max_idx = np.argmax(distance)
                    if distance[max_idx] > self.Threshold:
                        candidates[idx].append([ID[:-1], distance[max_idx]])
                        
        ID_Name = []
        # print('candidates {} | shape {}'.format(candidates, np.array(candidates).shape))
        print('Candidates')
        for cand in candidates:
            print(cand)
        for i in range(len(feature)):
            if len(candidates[i]) <= 0:
                ID_Name.append(None)
            else:
                max_idx = np.argmax([x[1] for x in candidates[i]])
                ID_Name.append(candidates[i][max_idx][0])

	return ID_Name

    def drawID(self, image, bboxes, ids):

        return

    def Verifier(self, faces, keypoints, save_id=False, ID=None):
	'''
	Args:
	faces: a list of cv2 images
        save_id: whether to save the feature to dataset
	ID: Name of this face , for saving to database 
	Return:
	a scalar in (0-1) to measure the similarity between face1 and face2 
	'''
        if save_id :
            if ID is None:
                print('input ID is empty')
                return False
            if len(ID) != len(faces):
                print('Number of IDs and Number of faces do not match')
                return False
        since = time.time()
        # perform face alignment
        origin_length = len(faces)
        # faces = [np.zeros((112, 96, 3)) for f in faces if f.shape[0] == 0 or f.shape[1] == 0]
        # for f_idx in range(origin_length):
            # if np.array(faces[f_idx]).shape[0] == 0 or np.array(faces[f_idx]).shape[1] == 0:
            #     faces[f_idx] = np.zeros((112, 96, 3))
        print('original length {}'.format(origin_length))
        faces = [alignment(faces[i], keypoints[i]) for i in range(origin_length)]
        #cv2.imshow('aligned', cv2.resize(faces[0], (96, 112)))
        if origin_length < cfg.max_face:
            for _ in range(cfg.max_face - origin_length):
                faces.append(np.zeros((112, 96, 3)))
        # resize to 112x96, using default mothod 

        faces = np.array([(cv2.resize(x, (96, 112)) - 127.5) / 128.0 for x in faces])
        faces = faces.transpose(0, 3, 1, 2)
	# TODO : here should be take carefully 
        ftrs = self.net.forward(data=faces)['fc5'][:origin_length]
	# if ID==null, we search the database to do verification 
	# else we save the ID and corresponding feature vector
	if save_id:
            for _ftr, _id in zip(ftrs, ID):
                ID_bucket = os.path.join(self.database_root, _id+'.npy')
                if os.path.exists(ID_bucket):
                    prev_ftrs = np.load(ID_bucket)
                    prev_ftrs = np.append(prev_ftrs, [_ftr], axis=0)
                    np.save(ID_bucket, prev_ftrs)
                else:
	            np.save(ID_bucket, [_ftr])

	        # add ID to ID_List
                if self.ID_List.count(_id+'\n') == 0:
                    with open(self.ID_path, 'a') as f:
                        f.write(_id+'\n')
                    with open(self.ID_path, 'r') as f:
                        self.ID_List = f.readlines()
	    return True 
	else:
            if len(ftrs.shape) <= 1:
                ftrs = ftrs[np.newaxis, :]
	    ID_Name = self.getid(ftrs)
            return ID_Name[:origin_length]

def UNIT():
	caffe.set_mode_gpu()
	caffe.set_device(0)
	model_proto = '../models/verifier/verifier.prototxt'
	model_weight = '../models/verifier/verifier.caffemodel'
	verify = Verifier(model_proto, model_weight)
        face_dict = {'Deng Wei':'deng_wei.jpg', 
                     'Deng Wei':'deng_wei-2.jpg', 
                     'Shi Yigong':'shi_yigong.jpg',
                     'Xue Qikun':'xue_qikun.jpg',
                     'Xue Qikun':'xue_qikun-2.jpg',
                     'Zhang Mu':'zhang_mu.jpg',
                     'Wang Mingzhi':'wang_mingzhi.jpg'}
        for key, value in face_dict.items():
            print(os.path.join('headteachers', value))
            img = cv2.imread(os.path.join('headteachers', value))
            status = verify.Verifier([img], save_id=True, ID=key)
            if status:
                print('Identity addition succeed')
            else:
                print('Identity addition failed')
        query = cv2.imread(os.path.join('headteachers', 'xue_qikun-3.jpg'))
        ids = verify.Verifier([query])
	print(ids)


def TEST_DIST():
    ftr1 = np.random.random((1,128));
    ftr2 = np.random.random((1,128));
    ans2 = ftr1.dot(ftr2.transpose()) /(np.sqrt(ftr1.dot(ftr1.transpose())) * np.sqrt(ftr2.dot(ftr2.transpose())) + 1.0e-5);
    ftr1 = np.expand_dims(ftr1 / np.sqrt(np.sum(ftr1 ** 2, 1, keepdims=True)), 1)
    ftr2 = np.expand_dims(ftr2 / np.sqrt(np.sum(ftr2 ** 2, 1, keepdims=True)), 0);
    
    ans1 = np.sum(ftr1 * ftr2, 2)
    print(ans1,ans2);

if __name__ == '__main__':
	TEST_DIST()
