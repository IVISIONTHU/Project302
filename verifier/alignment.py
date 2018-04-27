from matlab_cp2tform import get_similarity_transform_for_cv2
import numpy as np
import cv2

def alignment(src_img, src_pts):
    of = 0
    ref_pts = [[30.2946+of, 51.6963+of], [65.5318+of, 51.5014+of],
               [48.0252+of, 71.7366+of], 
               [33.5493+of, 92.3655+of], [62.7299+of, 92.2041+of]]
    crop_size = (96 + of * 2, 112 + of * 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img
