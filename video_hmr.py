from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from tools import visualize, preprocess_image
def del_all_flags(FLAGS):
    FLAGS.remove_flag_values(FLAGS.flag_values_dict())

del_all_flags(tf.flags.FLAGS)
import src.config
from src.RunModel import RunModel
from src.util import renderer as vis_util

flags.DEFINE_string('img_path', 'hmr/data', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def main(img_path, json_path=None, viz=True, renderer=None, config=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    cropped_imgs, params, og_imgs = preprocess_image(img_path, config.img_size, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_imgs = [np.expand_dims(input_img, 0) for input_img in cropped_imgs]

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    for k in range(len(input_imgs)):
        joints, verts, cams, joints3d, theta = model.predict(
            input_imgs[k], get_theta=True)
        print(joints.shape)
        print(verts.shape)
        print(cams.shape)
        print(joints3d.shape)
        print(theta.shape)
        if viz:
            visualize(og_imgs[k], params[k], joints[0], verts[0], cams[0],renderer)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    main(config.img_path, config=config, renderer=renderer)
