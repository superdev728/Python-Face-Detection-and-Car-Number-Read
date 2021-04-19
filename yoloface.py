import argparse
import sys
import os
import cv2
import numpy as np
from utils import *
from plate_recognition_image import *

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model'),
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif', 'webp']

file_names = [fn for fn in os.listdir('images/')
              if any(fn.endswith(ext) for ext in included_extensions)]

def _main():

    output_file = ''
    for file_name in file_names:
        filename = 'images/'+file_name
        cap = cv2.VideoCapture(filename)
        output_file = file_name.split('.')[0] + '_yoloface.' + file_name.split('.')[1]
        
        has_frame, frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], swapRB=True, crop=False)

        net.setInput(blob)
        outs = net.forward(get_outputs_names(net))
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        recognition_plate_number(filename, frame)
        save_state = cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
        if save_state :
            print(f"{output_file} successfuly saved!!!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _main()
