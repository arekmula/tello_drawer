import cv2
from argparse import ArgumentParser

from image_processing import ImageProcessor
from tello import Tello


def development_main(image_source, args):
    if image_source == "built_camera":
        cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.filepath)

    image_processor = ImageProcessor()

    while cap.isOpened():
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # Exit if q pressed
                cv2.destroyAllWindows()
                break

            ret, frame = cap.read()
            if ret:
                image_resize_drawed, path_img, finish_drawing, drawing_points = image_processor.process_img(frame)
                frame_and_path = cv2.hconcat([image_resize_drawed, path_img])
                if finish_drawing:
                    cv2.imshow("frame", frame_and_path)
                    key = cv2.waitKey(0)
                    break

                cv2.imshow("frame", frame_and_path)
            else:
                break
        break

    cap.release()
    cv2.destroyAllWindows()


def tello_main(args):
    tello = Tello(local_ip=args.local_ip, local_port=args.local_port)
    image_processor = ImageProcessor()

    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Exit if q pressed
            cv2.destroyAllWindows()
            break

        frame = tello.read()
        if frame is not None:
            # The image received from tello is RGB, OpenCV works in BGR format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            image_resize_drawed, path_img, finish_drawing, drawing_points = image_processor.process_img(frame)
            frame_and_path = cv2.hconcat([image_resize_drawed, path_img])
            if finish_drawing:
                cv2.imshow("frame", frame_and_path)
                key = cv2.waitKey(0)
                break
            cv2.imshow("frame", frame_and_path)


def main(args):
    image_source = args.image_source

    if image_source == "built_camera" or image_source == "saved_file":
        development_main(image_source=image_source, args=args)
    else:
        tello_main(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image_source", metavar="image_source", type=str, default="built_camera",
                        choices=["built_camera", "saved_file", "tello"])
    args, _ = parser.parse_known_args()
    if args.image_source == "saved_file":
        parser.add_argument("--filepath", metavar="filepath", type=str, required=True)
    elif args.image_source == "built_camera":
        parser.add_argument("--camera_index", metavar="camera_index", type=int, default=0)
    elif args.image_source == "tello":
        parser.add_argument("--local_ip", metavar="local_ip", type=str, default="0.0.0.0")
        parser.add_argument("--local_port", metavar="local_port", type=int, default=8889)

    args, _ = parser.parse_known_args()

    main(args)
