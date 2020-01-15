import numpy as np 
import deepdish as dd
from scipy.spatial.transform import Rotation as R
import os
import glob
import matplotlib.pyplot as plt
from hmr.src.util.renderer import draw_bbox
import cv2

from process_deepmimic_humanoid import smpl_OneD_names, smpl_names_map
from process_deepmimic_humanoid import smpl_kintree_table
from process_deepmimic_humanoid import dm_joint_parents_id, dm_name_to_id
from process_deepmimic_humanoid import dm_joints_info

results_dir = "refined"
base_name = 'walken-1'
file_name = '%s.h5' % base_name
file_path = os.path.join(os.getcwd(), results_dir, file_name)

video_frames = list(sorted(glob.glob("out/%s/frames/*.jpg" % base_name)))

data = dd.io.load(file_path)



## coco joints list
## we don't loop over these, but they're used as a lookup map
## for when we loop over the DeepMimic guy's bits
model_joints_map = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear"
]


theta_wanted = list(smpl_names_map.keys())

## One-dimensional angles
oneD_theta = smpl_OneD_names

# "RightJoints": [3, 4, 5, 6, 7, 8],
# "LeftJoints": [9, 10, 11, 12, 13, 14],

# time (1), root pos(3), root orient(4), chest orient(4), neck orient(4),
# r.hip orient(4), r.knee orient(1), r.ankle(4), r.shoulder orient(4), r.elbow orient(1),
# l.hip orient(4), l.knee orient (1), l.ankle(4), l.shoulder orient(4), l.elbow orient(1)


json_mimic = {
    "Loop": "wrap",
    "Frames": []
}

FRAME_DURATION = round(1 / 20, 6)

def match_coco_joint(pb_name, joints3d):
    """From PB joint name and Coco (HMR) joints3d, get
    the joint position that PB example expects."""
    try:
        # easy case
        j3d_id = model_joints_map.index(pb_name)
        print("  - found index #%d (%s) in joints vec" % (j3d_id, model_joints_map[j3d_id]))
        return joints3d[j3d_id]
    except ValueError:
        print("  - [W]", pb_name, "is not a Coco joint")
        if pb_name == "chest":
            ## HMR output REMOVES the chest from the SMPL joints3d which is
            ## fucking terrible
            return (.2*joints3d[2] + .2*joints3d[3] +
                    .6 * joints3d[12])
        if pb_name == "eye":
            print("  - averaging eye positions")
            # only 1 eye in PB example: just get average
            return .5 * (joints3d[15] + joints3d[16])
        raise

from inverse_kinematics import get_angle, get_quaternion
from inverse_kinematics import PB_joint_info

pb_joint_names = PB_joint_info['joint_name']


pb_joint_frames = []

for s, item in data.items():
    item = item[0]  # get actual data
    s_int = int(s)
    
    thetas = np.asarray(item['theta']).astype(float)
    joints = np.asarray(item['joints'])[0].astype(float)
    joints3d = np.asarray(item['joints3d']).astype(float)
    
    cam = thetas[0, :3]
    poses = thetas[:, 3:75]
    shape_smpl = thetas[:, 75:]
    
    poses_reshaped = poses.reshape(24, 3)
    rot = R.from_rotvec(poses_reshaped)  # rot[0] is the global rotation
    
    X_ROTATE = R.from_euler("y", -90, degrees=True)
    X_ROT_MAT = X_ROTATE.as_matrix()
    # rot = rot * X_ROTATE
    poses_quats = rot.as_quat()
    
    joints3d[:, 1] *= -1
    joints3d[:, 2] *= -1
    
    
    cam_scales = cam[0]
    cam_transl = cam[1:]
    proc_params = item['proc_param']
    img_size = proc_params['target_size']  # processed image size
    start_pt = proc_params['start_pt']
    inv_proc_scale = 1./np.asarray(proc_params['scale'])
    bbox = proc_params['bbox']  # bbox is obtained from OpenPose: bbox here is (cx, cy, scale, x, y, h, w)
    
    
    principal_pt = np.array([img_size, img_size]) / 2.
    flength = 500.
    tz = flength / (0.5 * img_size * cam_scales)
    trans = np.hstack([cam_transl, tz])  # camera translation vector ??
    final_principal_pt = (principal_pt + start_pt) * inv_proc_scale
    kp_original = ((joints + 1) * 0.5) * img_size  # in padded image.
    kp_original = (kp_original + start_pt) * inv_proc_scale  # should be good
    
    trans[1] *= -1
    trans[2] *= -1
    
    
    cx, cy = bbox[[0, 1]].astype(int)
    root_pos = .5 * (joints3d[2] + joints3d[3])
    root_pos_2d = .5 * (joints[2] + joints[3])  # left hip
    root_rot = poses_quats[0]
    
    
    
    DEBUG_FRAMES = [  ]
    pplot = s_int in DEBUG_FRAMES
    
    
    if pplot:
        print("Debugging frame %s" % video_frames[s_int])
        frame = cv2.imread(video_frames[s_int])
        ## Lifted from render_original in HMR utils
        plt.figure()
        cv2.circle(frame, (cx, cy), 4, (0,0,200), thickness=10)  # BBOX CENTER
        orig_root_kp = .5 * (kp_original[2] + kp_original[3])  # root kp
        cv2.circle(frame, tuple(orig_root_kp.astype(int)), 3, (200,0,0), thickness=6)  # show the kp
        
        SHOW_JT_IDX = range(1,18)
        for idx in SHOW_JT_IDX:
            orig_kp = kp_original[idx]  # some joint
            cv2.circle(frame, tuple(orig_kp.astype(int)), 2, (0,100,200), thickness=6)  # show the kp
        draw_bbox(frame, bbox[-4:])
        plt.imshow(frame[:, :, ::-1])
        plt.show()
    
    ## Loop over the names we want
    ## and fill in frame information for PB inverse_kinematics
    _new_frame_info = []
    
    # for k, name in enumerate(theta_wanted):
    # loop over joints
    for k, pb_name in enumerate(PB_joint_info['joint_name']):
        if pb_name == 'root':
            _new_frame_info.append(root_pos)
            continue  # already filled in root_pos
        # k_idx = theta_names.index(name)  # idx of the theta
        # dm_name = smpl_names_map[name]  # unneeded ?
        # dm_id = dm_name_to_id[dm_name]
        #if s_int==0:
        # print("Filling in DM_ID #%d: %s" % (dm_id, dm_name))
        print("Filling in PBID #%d: %s" % (k, pb_name))
        # print("  - got index #%d for SMPL %s" % (k_idx, theta_names[k_idx]))
        
        ## BUILD ROTATIONS FROM COORDINATES
        # grab joint coordinate for k_idx
        coord = match_coco_joint(pb_name, joints3d)
        _new_frame_info.append(coord)
    print()
    pb_joint_frames.append(_new_frame_info)


import json
from inverse_kinematics import coord_seq_to_rot_seq

rotation_sequence = coord_seq_to_rot_seq(pb_joint_frames, FRAME_DURATION)

json_mimic = {
    "Loop": "wrap",
    "Frames": rotation_sequence
}


output_filename = '%s_pb_mimicfile.json' % base_name
with open(output_filename, 'w') as f:
    json.dump(json_mimic, f, indent=4)

os.system("cp %s ../DeepMimic/walken_est_pb.json" % output_filename)