import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from timeit import default_timer as timer

import chainer
from chainercv.datasets import voc_bbox_label_names, voc_semantic_segmentation_label_colors
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox
from chainer import iterators

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.links import FeaturePredictor
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50
from chainercv.links import VGG16

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook


import pyrealsense2 as rs
import caffe

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

LIGHTWEIGHT_COEF = 1 # this may have no mean depending on the environment

def main():
    model_phase1 = YOLOv3(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='voc0712')


    chainer.cuda.get_device_from_id(0).use()
    model_phase1.to_gpu()

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    frame_count = 1
    while True:
        fframes = pipeline.wait_for_frames()
        color_frame = fframes.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())

        # BGR -> RGB
        rgb = cv2.cvtColor(cv2.resize(frame, (int(frame.shape[1] / LIGHTWEIGHT_COEF), int(frame.shape[0] / LIGHTWEIGHT_COEF))), cv2.COLOR_BGR2RGB)
        result = frame.copy()
        img = np.asarray(rgb, dtype=np.float32).transpose((2, 0, 1))

        bboxes, labels, scores = model_phase1.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        if len(bbox) != 0:
            for i, bb in enumerate(bbox):
                lb = label[i]
                if lb != 14:
                    continue
                conf = score[i].tolist()
                ymin = int(bb[0] * LIGHTWEIGHT_COEF)
                xmin = int(bb[1] * LIGHTWEIGHT_COEF)
                ymax = int(bb[2] * LIGHTWEIGHT_COEF)
                xmax = int(bb[3] * LIGHTWEIGHT_COEF)

                class_num = int(lb)
                cv2.rectangle(result, (xmin, ymin), (xmax, ymax), voc_semantic_segmentation_label_colors[class_num], 2)

                text = "person"
                print(text)

                text_top = (xmin, ymin - 10)
                text_bot = (xmin + 80, ymin + 5)
                text_pos = (xmin + 5, ymin)

                # Draw label 1
                cv2.rectangle(result, text_top, text_bot,
                              voc_semantic_segmentation_label_colors[class_num], -1)
                cv2.putText(result, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS:" + str(curr_fps)
            curr_fps = 0

        # Draw FPS in top right corner
        cv2.rectangle(result, (590, 0), (640, 17), (0, 0, 0), -1)
        cv2.putText(result, fps, (595, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Draw Frame Number
        cv2.rectangle(result, (0, 0), (50, 17), (0, 0, 0), -1)
        cv2.putText(result, str(frame_count), (0, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Output Result
        cv2.imshow("Yolo Result", result)

        # Stop Processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1


if __name__ == '__main__':
    main()