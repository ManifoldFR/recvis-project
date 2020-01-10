import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import os
import sys
import json
from tqdm import tqdm
import time
from fn import getTime

from pPose_nms import pose_nms, write_json

def correct_json_save(json_dir):
    for jfile in os.listdir(json_dir):
        if jfile[-5:] == ".json":
            with open(os.path.join(json_dir,jfile)) as json_file:
                data = json.load(json_file)
                for key in data:
                    json.dump(data[key], open(os.path.join(json_dir, key[:-4] + '.json'), 'w'))
            os.remove(os.path.join(json_dir,jfile))

def call_alphapose(input_dir,output_dir,format='open',batchSize=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for root, dirs, files in os.walk(input_dir):
        im_names = files
    print(files)
    data_loader = ImageLoader(im_names, batchSize=batchSize, format='yolo', dir_path=input_dir).start()
    det_loader = DetectionLoader(data_loader, batchSize=batchSize).start()
    det_processor = DetectionProcessor(det_loader).start()   
    # Load pose model
    pose_dataset = Mscoco()
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    # Init data writer
    writer = DataWriter(False).start()
    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation
            
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, output_dir, _format=format)
    correct_json_save(output_dir)
    print('Over')

#call_alphapose('/home/jules/Documents/recvis-project/AlphaPose/examples/demo','/home/jules/Documents/recvis-project/AlphaPose/examples/res')
