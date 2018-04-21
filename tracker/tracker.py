import caffe 
import numpy as np
import cv2
class Tracker:
	def __init__(self,model_proto,model_weight,CONF_THRESH = 0.8,NMS_THRESH = 0.3 ,FACTOR = 2):
		self.net = caffe.Net(model_proto,model_weight,caffe.TEST);
		self.conf_thresh = CONF_THRESH;
		self.nms_thresh = NMS_THRESH;

		self.bbox_cache = np.array([]);
		self.image_cache = np.array([]);

		self.factor = FACTOR;
		self.scale_factor = 10;
		# Problems here 
		self.mean = np.array([104,117,123]);
		# self.mean = np.reshape(self.mean,(1,1,3));
		self.image_dims = 0;
		# record the edge_spacing info
		self.edge_space = [];
		# record the padding_location info
		self.pad_location = [];
		
		self.transformer = 'caffe.io.transformer';
		self.SetupNetwork();
	def SetupNetwork(self):

		in_ = self.net.inputs[0];
		
		self.transformer = caffe.io.Transformer({in_:self.net.blobs[in_].data.shape});
		# Substract mean value 
		self.transformer.set_mean(in_, self.mean);
		# image scale -> 0-255
		self.transformer.set_raw_scale(in_, 255);
		
		self.image_dims = np.array(self.net.blobs[in_].data.shape[2:]);
		
		self.transformer.set_transpose(in_, (2, 0, 1))
	
	def Postprocess(self, bboxes,search_region):
		for index in range(bboxes.shape[0]):
			bboxes[index]  = bboxes[index] / self.scale_factor;

			bboxes[index][0] = bboxes[index][0] * search_region[index][0];
			bboxes[index][1] = bboxes[index][1] * search_region[index][0];
			bboxes[index][2] = bboxes[index][2] * search_region[index][1];
			bboxes[index][3] = bboxes[index][3] * search_region[index][1];
			
			bboxes[index][0] = max(0.0,bboxes[index][0] + self.pad_location[index][0] - self.edge_space[index][0]);
 
			bboxes[index][1] = max(0.0,bboxes[index][1] + self.pad_location[index][1] - self.edge_space[index][1]);

			bboxes[index][2] = min(self.image_dims[1],bboxes[index][2] + self.pad_location[index][0] - self.edge_space[index][0]);

			bboxes[index][3] = min(self.image_dims[1],bboxes[index][2] + self.pad_location[index][1] - self.edge_space[index][1]);

			return bboxes;
		
	def ConfFilter(self,dets,CONF_THRESH):
		pass
	def UpdateBBoxCache(self,bboxes):
		self.bbox_cache = bboxes;
	def UpdateImageCache(self,image):
		self.image_cache = image; 
	def GetBoxCenter(self,bbox):
		'''
		return center point location of a bbox (x1,y1) 
		'''
		return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0;
	def GetOutputSize(self,bbox):
		return self.factor * (bbox[2] - bbox[0]),self.factor * (bbox[3] - bbox[1]);
	def ComputeCropPadImageLocation(self,bbox,image):
		'''
		Args: 
		input: bbox np.array  image: cv2 image 
		return crop area location np.array
		'''
		# get image info 
		bbox_center_x , bbox_center_y = self.GetBoxCenter(bbox);
		image_width = image.shape[0];
		image_height = image.shape[1];
		output_width , output_height = self.GetOutputSize(bbox);
		
		roi_left = max(0.0,bbox_center_x  - output_width / 2.0);
		roi_bottom = max(0.0,bbox_center_y -output_height / 2.0);
		# compute roi width
		left_half = min(output_width / 2, bbox_center_x);
		right_half = min(output_width / 2, image_width - bbox_center_x);
		roi_width = max(1.0,left_half + right_half);
		# compute roi height
		top_half = min(output_height / 2, bbox_center_y);
		bottom_half = min(output_height / 2,image_height - bbox_center_y);
		roi_height = max(1.0,top_half + bottom_half);

		
		# return roi_left,roi_bottom,roi_left + roi_width,roi_bottom + roi_height;
		return int(roi_left), int(roi_bottom), int(roi_width), int(roi_height);


		
	# Crop a region from image descripted by bbox ,and do padding when necessary
	def CropPadImage(self,bbox,image):
		# some basic information for image 
		edge_spacing_x = edge_spacing_y = 0;
		output_width,output_height = self.GetOutputSize(bbox);
		bbox_center_x ,bbox_center_y = self.GetBoxCenter(bbox);
		image_width = image.shape[0];
		image_height = image.shape[1];
		# get roi 
		# x1,y1,x2,y2 = self.ComputeCropPadImageLocation(bbox,image);
		roi_left, roi_bottom, roi_width, roi_height = self.ComputeCropPadImageLocation(bbox,image);
		x1, x2 = roi_left, roi_left + roi_width
		y1, y2 = roi_bottom, roi_bottom + roi_height

		pad_location = np.array([x1,y1,x2,y2]);
		self.pad_location.append(pad_location);

		roi_left = int(min(x1,image_width - 1));
		roi_height = int(min(image_height,max(1.0,y2 - y1)));
		
		cropped_image = image[int(roi_bottom):int(roi_bottom + roi_height),int(roi_left):int(roi_left + roi_width),:]
		print('cropped image size = {}'.format(cropped_image.shape));

		output_width = int(max(np.ceil(output_width),roi_width));
		output_height = int(max(np.ceil(output_height),roi_height));

		output_image = np.zeros((output_height,output_width,3),np.float);
		print('output_image size = {}'.format(output_image.shape));

		output_image[:,:,:] = 0;
		
		edge_spacing_x = int(max(0.0,output_width / 2.0 - bbox_center_x));
		edge_spacing_y = int(max(0.0,output_height / 2.0 - bbox_center_y));

		edge_spacing_x = min(edge_spacing_x, image_width - 1);
		edge_spacing_y = min(edge_spacing_y, image_height - 1);
		
		edge_space = np.array([edge_spacing_x,edge_spacing_y]);
		self.edge_space.append(edge_space);

		output_image[edge_spacing_y : edge_spacing_y + roi_height ,edge_spacing_x : edge_spacing_x + roi_width ,:] = cropped_image;
		return output_image.copy();
	
	

	def GetInput(self, inputs):
		input_target = np.zeros((len(self.bbox_cache),
					self.image_dims[0],
					self.image_dims[1],
					inputs.shape[2]
					),
					dtype = np.float32);
		input_image = np.zeros((len(self.bbox_cache),
					self.image_dims[0],
					self.image_dims[1],
					inputs.shape[2]
					),
					dtype = np.float32);
		# store the search region 
		search_region = np.zeros((len(self.bbox_cache),
					2
					),
					dtype = np.float32);
	
		for index in range(len(self.bbox_cache)):
			target = self.CropPadImage(self.bbox_cache[index],self.image_cache);
			search_region[index][0],search_region[index][1] = target.shape[0],target.shape[1];
			target = caffe.io.resize_image(target,self.image_dims);
			input_target[index] = target.copy();

			image = self.CropPadImage(self.bbox_cache[index],inputs);
			image = caffe.io.resize_image(image,self.image_dims);
			input_image[index] = image.copy();

		caffe_target = np.zeros(np.array(input_target.shape)[[0,3,1,2]],dtype = np.float32);
		caffe_image = np.zeros(np.array(input_image.shape)[[0,3,1,2]],dtype = np.float32);
		for index in range(caffe_target.shape[0]):
			# input_target[index] = caffe.io.resize_image(input_target[index], self.image_dims)
			# input_image[index] = caffe.io.resize_image(input_image[index], self.image_dims)
			caffe_target[index] = self.transformer.preprocess(self.net.inputs[0], input_target[index].copy())
			caffe_image[index] = self.transformer.preprocess(self.net.inputs[0], input_image[index].copy())
		forward_kwargs = {'target':caffe_target, 'image':np.tile(np.transpose(image, (2, 0, 1))[np.newaxis, :], (len(self.bbox_cache), 1, 1, 1))}
		# forward_kwargs = {'target':caffe_target, 'image':image}
		return forward_kwargs, search_region
			  
	# DO Track
	def Track(self, image):
		'''
		This is a tracker interface for tracking objects in image
		and it returns bboxes of tracked objects 
		input : a cv2 image ,i.e. BGR for channels
		return : bboxes  
	
		'''
		if len(self.bbox_cache) < 1:
			return np.array([])
		# prepare for the input
		forward_kwargs, search_region = self.GetInput(image)
		# reshape the net
		forward_kwargs['bbox'] = np.zeros((len(self.bbox_cache), 4, 1, 1))
		self.net.blobs['target'].reshape(*(forward_kwargs['target'].shape))
		self.net.blobs['image'].reshape(*(forward_kwargs['image'].shape))
		self.net.blobs['bbox'].reshape(*(forward_kwargs['bbox'].shape))
		self.net.forward(**forward_kwargs)
		bboxes = self.net.blobs['fc8'].data;
		# post process for bboxes 
		bboxes = self.Postprocess(bboxes, search_region);
		# set cache and clear cache
		self.image_cache = image;
		self.bbox_cache = bboxes;
		self.edge_space = [];
		self.pad_location = [];
		# return the result 
		return bboxes

