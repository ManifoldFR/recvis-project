import json, os, glob
import numpy as np
import pandas as pd 

def join_csv():
    path = 'bvh/csv/'                   
    all_files = glob.glob(os.path.join(path, "*.csv"))
    all_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

    concatenated_df['frame'] = concatenated_df.index+1
    concatenated_df.to_csv("bvh/csv_joined.csv", index=False)

file_name = 'refined/hammer.json'

with open(os.path.join(os.getcwd(), file_name)) as f:
    data = json.load(f)

keys = [int(k) for k in data]
keys = sorted(keys)

joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
            'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
            'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
            'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
            'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
            'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
            'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
            'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
            'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
            'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
            'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
            'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
            'Neck_x', 'Neck_y', 'Neck_z', 
            'Head_x', 'Head_y', 'Head_z', 
            'Nose_x', 'Nose_y', 'Nose_z', 
            'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
            'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
            'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
            'Ear.R_x', 'Ear.R_y', 'Ear.R_z']

for k in keys:
    print("Frame %d: joints3d shape %s" % (k, str(np.array(data[str(k)]['joints3d']).shape)))
    joints_export = pd.DataFrame(np.array(data[str(k)]['joints3d']).reshape(1,57), columns=joints_names)
    joints_export.drop(['Nose_x', 'Nose_y', 'Nose_z', 
            'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
            'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
            'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
            'Ear.R_x', 'Ear.R_y', 'Ear.R_z'],axis=1)
    joints_export.index.name = 'frame'
    joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
    joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1
    
    hipCenter = joints_export.loc[:][['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                      'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]

    joints_export['hip.Center_x'] = hipCenter.iloc[0][::3].sum()/2
    joints_export['hip.Center_y'] = hipCenter.iloc[0][1::3].sum()/2
    joints_export['hip.Center_z'] = hipCenter.iloc[0][2::3].sum()/2
    
    os.makedirs("bvh/csv/", exist_ok=True)
    joints_export.to_csv("bvh/csv/%06d.jpg.csv" % k)

join_csv()
