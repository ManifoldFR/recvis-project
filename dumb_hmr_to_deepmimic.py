import numpy as np 
import deepdish as dd
from scipy.spatial.transform import Rotation as R
import os
import glob
import matplotlib.pyplot as plt
from hmr.src.util.renderer import draw_bbox
import cv2

results_dir = "refined"
base_name = 'walken-1'
file_name = '%s.h5' % base_name
file_path = os.path.join(os.getcwd(), results_dir, file_name)

video_frames = list(sorted(glob.glob("out/%s/frames/*.jpg" % base_name)))
print("Video frames:", video_frames)

data = dd.io.load(file_path)


def change_camera(camera,point):
    return (point[:2] + camera[1:]) * camera[0]



theta_names = [
                'Left_Hip',
                'Right_Hip', 
                'Waist', 
                'Left_Knee', 
                'Right_Knee',
                'Upper_Waist', 
                'Left_Ankle', 
                'Right_Ankle', 
                'Chest',
                'Left_Toe', 
                'Right_Toe', 
                'Base_Neck', 
                'Left_Shoulder',
                'Right_Shoulder', 
                'Upper_Neck', 
                'Left_Arm', 
                'Right_Arm',
                'Left_Elbow', 
                'Right_Elbow', 
                'Left_Wrist', 
                'Right_Wrist',
                'Left_Finger', 
                'Right_Finger'
            ]

theta_wanted = [
                'Chest',
                'Upper_Neck', 
                'Right_Hip', 
                'Right_Knee',
                'Right_Ankle', 
                'Right_Arm', 
                'Right_Elbow', 
                #'Right_Wrist',
                'Left_Hip',
                'Left_Knee', 
                'Left_Ankle', 
                'Left_Arm',
                'Left_Elbow', 
                #'Left_Wrist', 
            ]

## One-dimensional angles
oneD_theta = ['Right_Knee','Right_Elbow','Left_Knee','Left_Elbow']

root = "Waist"
# root = "Left_Hip"

# "RightJoints": [3, 4, 5, 6, 7, 8],
# "LeftJoints": [9, 10, 11, 12, 13, 14],

# time (1), root pos(3), root orient(4), chest orient(4), neck orient(4),
# r.hip orient(4), r.knee orient(1), r.ankle(4), r.shoulder orient(4), r.elbow orient(1),
# l.hip orient(4), l.knee orient (1), l.ankle(4), l.shoulder orient(4), l.elbow orient(1)

json_mimic = {
    "Loop": "wrap",
    "Frames": []
}
    

for s, item in data.items():
    item = item[0]  # get actual data
    # output for frame k
    output = [ 1 / 24 ]  
    
    s_int = int(s)
    
    thetas = np.asarray(item['theta']).astype(float)
    joints = np.asarray(item['joints'])[0].astype(float)
    joints3d = np.asarray(item['joints3d']).astype(float)
    
    cam = thetas[0, :3]
    poses = thetas[:, 3:75]
    shape_smpl = thetas[:, 75:]
    
    cam_scales = cam[0]
    cam_transl = cam[1:]
    proc_params = item['proc_param']
    img_size = proc_params['target_size']  # processed image size
    start_pt = proc_params['start_pt']
    inv_proc_scale = 1./np.asarray(proc_params['scale'])
    bbox = proc_params['bbox']  # bbox is obtained from OpenPose: bbox here is (cx, cy, scale, x, y, h, w)
    
    
    pplot = False
    DEBUG_FRAMES = [10, 20, 30, 40]
    if s_int in DEBUG_FRAMES:
        pplot = True
    
    principal_pt = np.array([img_size, img_size]) / 2.
    flength = 500.
    tz = flength / (0.5 * img_size * cam_scales)
    trans = np.hstack([cam_transl, tz])  # camera translation vector ??
    final_principal_pt = (principal_pt + start_pt) * inv_proc_scale
    kp_original = ((joints + 1) * 0.5) * img_size  # in padded image.
    kp_original = (kp_original + start_pt) * inv_proc_scale  # should be good
    
    poses_reshaped = poses.reshape(24, 3)
    
    cx, cy = bbox[[0, 1]].astype(int)
    root_pos = .5 * (joints3d[2] + joints3d[3])
    root_pos_2d = .5 * (joints[2] + joints[3])  # left hip
    root_rot = [1.,0.,0.,0.]
    
    ## 
    root_pos += trans

    output += root_pos.tolist()
    output += root_rot
    ### INCORRECT ###
    
    
    if pplot:
        print("Debugging frame %s" % video_frames[s_int])
        frame = cv2.imread(video_frames[s_int])
        ## Lifted from render_original in HMR utils
        plt.figure()
        cv2.circle(frame, (cx, cy), 4, (0,0,200), thickness=10)  # BBOX CENTER
        orig_root_kp = .5 * (kp_original[2] + kp_original[3])  # root kp
        cv2.circle(frame, tuple(orig_root_kp.astype(int)), 3, (0,100,200), thickness=6)  # show the kp
        
        SHOW_JT_IDX = []
        for idx in SHOW_JT_IDX:
            orig_kp = kp_original[idx]  # some joint
            cv2.circle(frame, tuple(orig_kp.astype(int)), 3, (0,100,200), thickness=6)  # show the kp
        draw_bbox(frame, bbox[-4:])
        plt.imshow(frame[:, :, ::-1])
        plt.show()
        
    
    rot = R.from_rotvec(poses_reshaped)
    poses_quats = rot.as_quat()
    
    ## Loop over the names we want
    for k, name in enumerate(theta_wanted):
        if s_int==0:
            print("Filling in #%d: %s" % (k, name))
        k_idx = theta_names.index(name)  # idx of the theta
        if name in oneD_theta:
            angle = np.linalg.norm(poses_reshaped[k_idx])
            sign = -np.sign(poses_reshaped[k_idx][-1])
            output.append(sign*angle)
        else :
            r_ = poses_quats[k_idx]
            output += r_.tolist()
            
    # if name in oneD_theta:
    #     angle = np.linalg.norm(poses_reshaped[k_idx])
    #     sign = -np.sign(poses_reshaped[k_idx][-1])
    #     output.append(sign*angle)
    # else :
    #     r_ = poses_quats[k_idx]
    #     # Now compose r_ with the kinematic tree parents we remove
    #     if name == 'Upper_Waist':
    #         # DeepMimic guy has no waist/upper waist
    #         # Corresponding DM joint is 'chest'
    #         # r_ = r_ * poses_quats[theta_names.index('Upper_Waist')]
    #         # r_ = r_ * poses_quats[theta_names.index('Waist')]
    #     if name == 'Base_Neck':
    #         # DeepMimic guy has no base neck
    #         r_ = poses_quats[theta_names.index('Base_Neck')] * r_
    #     if name == 'Right_Shoulder':
    #         # r_ = poses_quats[theta_names.index('Right_Shoulder')] * r_
    #     if name == 'Left_Shoulder':
    #         # r_ = poses_quats[theta_names.index('Left_Shoulder')] * r_
    #     if name == 'Left_Hip' or name == 'Right_Hip':
    #         # DeepMimic guy has no waist or upper waist, just a chest
    #         # propagate back to the chest
    #         # r_ = r_ * poses_quats[theta_names.index('Waist')]
    #         # r_ = r_ * poses_quats[theta_names.index('Upper_Waist')]
    #     output += r_.tolist()
    
    json_mimic['Frames'].append(output)


import json

with open('%s_dumb_mimicfile.json' % base_name,'w') as f:
    json.dump(json_mimic, f, indent=4)

