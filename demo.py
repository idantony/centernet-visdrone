from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import cv2
from src.opts import opt
from src.detector import Detector
import numpy as np


def save_results(opt, image, results, path):
    for cls_ind in range(1, opt.dataset_info["num_classes"] + 1):
        for bbox in results[cls_ind]:
            conf = bbox[4]
            # filter low score
            if conf < opt.vis_thresh:
                continue
            bbox = np.array(bbox[:4], dtype=np.int32)

            class_name = opt.dataset_info["class_name"]

            cv2.rectangle(img=image,
                          pt1=(bbox[0], bbox[1]),
                          pt2=(bbox[2], bbox[3]),
                          color=[0, 255, 0],
                          thickness=1)
            #txt
            cv2.putText(img=image,
                        text=f'{class_name[cls_ind-1]}{"-"}{conf*100:.1f}',
                        org=(bbox[0], bbox[1] - 2),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    if path is not None and len(path) > 0:
        cv2.imwrite(path, image)
    return image


def demo():
    detector = Detector(opt)
    #image
    if opt.video is None or len(opt.video) <= 0:
        image_path = opt.image
        print("input image path:", image_path)
        image = cv2.imread(image_path)
        ret = detector.run(image)
        save_results(opt, image, ret['results'], 'demo_result.png')
        time_str = ''
        for stat in ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']:
            time_str += f'{stat} {ret[stat]:.3f}s |'
        print(time_str)
    #video
    else:
        print("input video path:", opt.video)
        capture = cv2.VideoCapture(opt.video)
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('FPS', fps)
        print('Size', size)
        writer = cv2.VideoWriter(opt.video_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
        ret, frame = capture.read()
        cv2.imshow('demo', frame)
        cv2.resizeWindow("demo", int(size[0]/2), int(size[1]/2))
        while True:
            print("start loop", datetime.datetime.now())
            ret, frame = capture.read()  # read each frame from the video
            if ret:
                print("start detect", datetime.datetime.now())
                detect_result = detector.run(frame)
                print("end detect", datetime.datetime.now())
                result_image = save_results(opt, frame, detect_result['results'], None)
                writer.write(result_image)
                scale_image = cv2.resize(result_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                cv2.imshow('demo', scale_image)  # real time display of each frame
                print("end loop", datetime.datetime.now())
            else:
                break
            if cv2.waitKey(1) == '27':
                break
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


        # def demo():
#     detector = Detector(opt)
#     image_path = opt.image
#     print("input image path:", image_path)
#     image = cv2.imread(image_path)
#
#     ret = detector.run(image)
#
#     save_results(opt, image, ret['results'], 'demo_result.png')
#
#     time_str = ''
#     for stat in ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']:
#         time_str += f'{stat} {ret[stat]:.3f}s |'
#     print(time_str)

if __name__ == '__main__':

    demo()
