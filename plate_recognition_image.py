#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import json
import time
from collections import OrderedDict
from glob import glob
import cv2
import requests
import os

def hasNumbers(inputString):
    state = False
    for character in inputString:
        if character.isdigit():
            state = True
    return state

def recognition_plate_number(path, frame):
    result = []
    with open(path, 'rb') as fp:
        response = requests.post(
                        'https://api.platerecognizer.com/v1/plate-reader/',
                        files=dict(upload=fp),
                        data=dict(regions='fr'),
                        headers={'Authorization': 'Token ' + '46569c6bbf83ec3257068d20a74113e420598687'})
    result.append(response.json(object_pairs_hook=OrderedDict))
    time.sleep(1)

    im=cv2.imread(path)
          
    resp_dict = json.loads(json.dumps(result, indent=2))

    for resp_dict_object in resp_dict[0]['results']:
        num=resp_dict_object['plate']
        num = num.upper()
        boxs=resp_dict_object['box']
        candidates = resp_dict_object['candidates']
        car_number = ""
        for candidate in candidates :
            is_num = hasNumbers(candidate['plate'])
            if candidate['score'] > 0.8 and is_num:
                car_number = candidate['plate']
            else :
                car_number = num
        num = car_number.upper()
        xmins, ymins, ymaxs, xmaxs=boxs['xmin'],boxs['ymin'],boxs['ymax'],boxs['xmax']

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if hasNumbers(num) :
            edges = cv2.Canny(img,150,200)
            cv2.rectangle(frame, (xmins, ymins), (xmaxs, ymaxs), (255,0,0), 2)
            cv2.rectangle(edges, (xmins, ymins), (xmaxs, ymaxs), (255,0,0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, num, (xmins, ymins), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
            cv2.destroyAllWindows()
