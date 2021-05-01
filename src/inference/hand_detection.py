#!/usr/bin/env python3


import cv2
from yolo import YOLO


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




if __name__ == '__main__':
    print("test hand_detection")
    hand_detector = HandDetector()
    hand_detector.predict()
