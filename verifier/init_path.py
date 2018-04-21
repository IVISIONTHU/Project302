
"""Set up paths for Our project."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path ='../caffe/python'; #osp.join(this_dir, '..','caffe', 'python')
add_path(caffe_path)

# Add detection  to PYTHONPATH
detector_path = osp.join(this_dir,'..','detector')
add_path(detector_path)

detector_libpath = osp.join(this_dir,'..','detector','lib')
add_path(detector_libpath)

# Add track  to PYTHONPATH
tracker_path = osp.join(this_dir,'..','tracker')
add_path(tracker_path)
# Add track  to PYTHONPATH
project_path = osp.join(this_dir,'..','project302')
add_path(project_path)

#caffe_path ='/data2/Project302/caffe/python'; #osp.join(this_dir, '..','caffe', 'python')
#add_path(caffe_path)

