import json

with open("../DeepMimic/data/characters/humanoid3d.txt") as f:
    data = json.load(f)

skeleton = data['Skeleton']
joints = skeleton['Joints']

# From SMPL names to their DeepMimic counterparts
smpl_names_map = {
    'Chest': 'chest',
    'Upper_Neck': 'neck', 
    'Right_Hip': 'right_hip', 
    'Right_Knee': 'right_knee',
    'Right_Ankle': 'right_ankle', 
    'Right_Arm': 'right_shoulder', 
    'Right_Elbow': 'right_elbow', 
    'Left_Hip': 'left_hip',
    'Left_Knee': 'left_knee', 
    'Left_Ankle': 'left_ankle', 
    'Left_Arm': 'left_shoulder',
    'Left_Elbow': 'left_elbow'
}
