
# coding: utf-8

# In[9]:


import os
import os.path as osp
import numpy as np
import argparse
import math
earth_radian = 6378137
degree_2_radian = 0.017453293


# In[2]:


parser = argparse.ArgumentParser(description='Figure out global pose')
parser.add_argument('--data_dir', type=str, default='/home/drinkingcoder/Dataset/sensetime/')
parser.add_argument('--scene', type=str, default='Museum')
args = parser.parse_args()


# In[3]:


process_list_path = osp.join(args.data_dir, args.scene, 'process_list.txt')
with open(process_list_path) as f:
    process_list = [line.rstrip('\n') for line in f]
print('process_list = {}'.format(process_list))


# In[4]:


seq = process_list[0]
oxts_path = osp.join(args.data_dir, args.scene, seq, 'OXTS.csv')
camera_times_path = osp.join(args.data_dir, args.scene, seq, 'times.txt')
global_pose_path = osp.join(args.data_dir, args.scene, seq, 'global_pose.txt')
print(oxts_path)
print(camera_times_path)
print(global_pose_path)
# In[20]:


def lla2xyz(lat, lon, alt):
    radian = math.cos(lat * degree_2_radian) * earth_radian
    east = radian * lon * degree_2_radian
    north = radian * math.log(math.tan((90 + lat) * degree_2_radian / 2.0))
    x = north
    y = east
    z = -alt
    return x, y, z


# In[22]:


from robotcar_sdk.interpolate_poses import interpolate_poses
from robotcar_sdk.transform import build_se3_transform
def pose_from_oxts(ts, oxts):
    sys_time, gps_time, lat, lon, alt, roll, pitch, yaw, other = oxts.split(',',8)
    xyzrpy = [lat, lon, alt, roll, pitch, yaw]
    xyzrpy = [float(v) for v in xyzrpy]
    xyzrpy[0], xyzrpy[1], xyzrpy[2] = lla2xyz(xyzrpy[0], xyzrpy[1], xyzrpy[2])
    pose = build_se3_transform(xyzrpy)
    return pose

# In[23]:


def concat_ts_and_pose(ts, pose):
    res = [ts]
    for r in range(3):
        for c in range(4):
            res.append(str(pose[r, c]))
    res = ' '.join(res)
    return res

# In[24]:




# In[25]:


for seq in process_list:
    oxts_path = osp.join(args.data_dir, args.scene, seq, 'OXTS.csv')
    camera_times_path = osp.join(args.data_dir, args.scene, seq, 'times.txt')
    global_pose_path = osp.join(args.data_dir, args.scene, seq, 'global_pose.txt')

    with open(oxts_path, 'r') as f:
        oxts = [line.rstrip('\n').lstrip(' ') for line in f][1:]
    oxts_ts = [line.split(',', 2)[1] for line in oxts] # get gps timestamp in each line
    oxts_ts = np.asarray([ts.replace(' ', '-').replace(':', '-').replace('.', '-') for ts in oxts_ts])
    with open(camera_times_path, 'r') as f:
        camera_ts = [line.rstrip('\n') for line in f]
    global_pose = []
    print(oxts_ts[0:20])
    for item in camera_ts:
        ts = item.split(' ')[1] # 2018-07-30-15-09-32-546
        idx = np.amax(np.where(oxts_ts < ts)) # find the index of maximum element that is smaller that ts
        print(idx)
        pose = pose_from_oxts(ts, oxts[idx])
        global_pose.append(concat_ts_and_pose(ts, pose))
    with open(global_pose_path, 'w') as f:
        f.write('\n'.join(global_pose))

