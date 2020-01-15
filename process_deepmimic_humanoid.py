import json

with open("../DeepMimic/data/characters/humanoid3d.txt") as f:
    data = json.load(f)

skeleton = data['Skeleton']
dm_joints_info = skeleton['Joints']

dm_joint_parents_id = [jo["Parent"] for jo in dm_joints_info]
dm_name_to_id = {jo["Name"]: jo["ID"] for jo in dm_joints_info}


# From SMPL names to their DeepMimic counterparts
smpl_names_map = {
    'Upper_Waist': 'chest',
    'Base_Neck': 'neck', 
    'Right_Hip': 'right_hip', 
    'Right_Knee': 'right_knee',
    'Right_Ankle': 'right_ankle', 
    'Right_Shoulder': 'right_shoulder', 
    'Right_Elbow': 'right_elbow', 
    'Left_Hip': 'left_hip',
    'Left_Knee': 'left_knee', 
    'Left_Ankle': 'left_ankle', 
    'Left_Shoulder': 'left_shoulder',
    'Left_Elbow': 'left_elbow'
}

smpl_OneD_names = ['Right_Knee','Right_Elbow','Left_Knee','Left_Elbow']

## 'Stolened' from the neutral_smpl_with_cocoplus_reg.pkl file
smpl_kintree_table = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13,
       14, 16, 17, 18, 19, 20, 21]
