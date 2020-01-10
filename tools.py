import sys, os
import numpy as np
import skimage.io as io

parentdir = os.path.dirname('hmr/')
sys.path.insert(0,parentdir) 
from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util

def visualize(img, proc_param, joints, verts, cam, renderer):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, target_size, json_path=None):
    crops = []
    params = []
    imgs = []
    for img_name in sorted(os.listdir(img_path)):
        if not img_name.endswith('.jpg'):
            continue
        img = io.imread(os.path.join(img_path,img_name))
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if json_path is None:
            if np.max(img.shape[:2]) != target_size:
                print('Resizing so the max image size is %d..' % target_size)
                scale = (float(target_size) / np.max(img.shape[:2]))
            else:
                scale = 1.
            center = np.round(np.array(img.shape[:2]) / 2).astype(int)
            # image center in (x,y)
            center = center[::-1]
        else:
            scale, center = op_util.get_bbox(os.path.join(json_path,img_name))

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                target_size)

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)
        crops.append(crop)
        params.append(proc_param)
        imgs.append(img)
    return crops, params, imgs