#!/usr/bin/env python3


import cv2
from yolo import YOLO
import sys
import tensorflow as tf
import numpy as np

class HandDetector:

    def __init__(self):
        print("loading yolo...")
        self.yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])


    def predict(self):
        test_img = cv2.imread('./data/arek/circle/0000075.png')

        # cap = cv2.VideoCapture('./data/ksz/VID_20210430_103031897.mp4') # Ksz 1
        # cap = cv2.VideoCapture('./data/ksz/VID_20210430_103314641.mp4') # Ksz 2
        # cap = cv2.VideoCapture('./data/Helm_Thora/vokoscreen-2021-04-30_17-22-59.mkv') # JA 1
        cap = cv2.VideoCapture('./data/Helm_Thora/vokoscreen-2021-04-30_17-24-12.mkv') # JA 2
        # cap = cv2.VideoCapture('./data/kuba/VID_20210501_114402.mp4') # Kuba 1
        # cap = cv2.VideoCapture('./data/kuba/VID_20210501_114442.mp4') # Kuba 2



        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                # Display the resulting frame

                # frame = cv2.resize(frame, (848, 640))

                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Frame', 848, 640)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (848, 640))

                width, height, inference_time, results = self.yolo.inference(frame)

                print(results)

                conf_sum = 0
                detection_count = 0

                for detection in results:
                    id, name, confidence, x, y, w, h = detection
                    cx = x + (w / 2)
                    cy = y + (h / 2)

                    conf_sum += confidence
                    detection_count += 1

                    # draw a bounding box rectangle and label on the image
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    text = "%s (%s)" % (name, round(confidence, 2))
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.25, color, 1)

                    print("%s with %s confidence" % (name, round(confidence, 2)))

                cv2.imshow('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break
        #
        # test_img_resze = cv2.resize(test_img, (848, 640))
        # # test_img_resze = cv2.cvtColor(test_img_resze, cv2.COLOR_BGR2RGB)
        # width, height, inference_time, results = self.yolo.inference(test_img_resze)
        # print(results)


        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 848, 640)
        #
        # conf_sum = 0
        # detection_count = 0
        #
        # for detection in results:
        #     id, name, confidence, x, y, w, h = detection
        #     cx = x + (w / 2)
        #     cy = y + (h / 2)
        #
        #     conf_sum += confidence
        #     detection_count += 1
        #
        #     # draw a bounding box rectangle and label on the image
        #     color = (0, 0, 255)
        #     cv2.rectangle(test_img_resze, (x, y), (x + w, y + h), color, 1)
        #     text = "%s (%s)" % (name, round(confidence, 2))
        #     cv2.putText(test_img_resze, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.25, color, 1)
        #
        #     print("%s with %s confidence" % (name, round(confidence, 2)))
        #
        #     # cv2.imwrite("export.jpg", mat)
        #
        # # show the output image
        # cv2.imshow('image', test_img_resze)
        # cv2.waitKey(1000)



        # cv2.imshow('img', test_img_resze)
        # cv2.waitKey(1000)


class HandDetector2:


    # MODEL_NAME = 'hand_inference_graph'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './models//frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './models/hand_label_map.txt'

    def __init__(self):
        print("init hand detector 2")

        self.detection_graph = tf.Graph()

        self.detection_graph, self.sess = self.load_inference_graph()

    # Load a frozen infrerence graph into memory
    def load_inference_graph(self):
        # load frozen tensorflow model into memory
        print("> ====== loading HAND frozen graph into memory")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=detection_graph)
        print(">  ====== Hand Inference graph loaded.")
        return detection_graph, self.sess

    # draw the detected bounding boxes on the images
    # You can modify this to also draw a label.
    def draw_box_on_image(self, num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
        for i in range(num_hands_detect):
            if (scores[i] > score_thresh):
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                              boxes[i][0] * im_height, boxes[i][2] * im_height)
                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))
                cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)


    # Actual detection .. generate scores and bounding boxes given an image
    def detect_objects(self, image_np, detection_graph, sess):
        # Definite input and output Tensors for detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [detection_boxes, detection_scores,
                detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return np.squeeze(boxes), np.squeeze(scores)

    def predict(self):

        test_img = cv2.imread('./data/arek/circle/0000176.png')

        # cap = cv2.VideoCapture('./data/ksz/VID_20210430_103031897.mp4') # Ksz 1
        # cap = cv2.VideoCapture('./data/ksz/VID_20210430_103314641.mp4') # Ksz 2
        cap = cv2.VideoCapture('./data/Helm_Thora/vokoscreen-2021-04-30_17-22-59.mkv') # JA 1
        # cap = cv2.VideoCapture('./data/Helm_Thora/vokoscreen-2021-04-30_17-24-12.mkv')  # JA 2
        # cap = cv2.VideoCapture('./data/kuba/VID_20210501_114402.mp4') # Kuba 1
        # cap = cv2.VideoCapture('./data/kuba/VID_20210501_114442.mp4') # Kuba 2

        if (cap.isOpened() == False):
            print("Error opening video stream or file")


        # max number of hands we want to detect/track
        num_hands_detect = 1

        cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, image_np = cap.read()
            # image_np = test_img
            if ret == True:

                try:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                except:
                    print("Error converting to RGB")



                image_np = cv2.resize(image_np, (320, 240))
                im_height, im_width = image_np.shape[:2]

                boxes, scores = self.detect_objects(image_np, self.detection_graph, self.sess)

                # draw bounding boxes on frame
                thresh = 0.5
                self.draw_box_on_image(num_hands_detect, thresh,
                                                 scores, boxes, im_width, im_height,
                                                 image_np)

                cv2.imshow('Single-Threaded Detection',
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                cv2.waitKey(10)
            # Break the loop
            else:
                break


if __name__ == '__main__':
    print("test hand_detection")
    # hand_detector = HandDetector()
    # hand_detector.predict()

    hand_detector2 = HandDetector2()
    hand_detector2.predict()

