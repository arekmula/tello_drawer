import cv2
import time
from argparse import ArgumentParser
from pathlib import Path

from tello import Tello


def main(args):
    tello = Tello(local_ip=args.local_ip, local_port=args.local_port)

    # Create directory to save images if it doesn't exists
    if args.save_img:
        timestamp = str(time.time())
        save_dir = Path(f"{args.save_dir}") / Path(timestamp)
        save_dir.mkdir(parents=True, exist_ok=True)

    fps_delay_ms = int((1 / args.fps) * 1000)

    save_frame_count = 0
    cv2.namedWindow("tello")
    while True:

        key = cv2.waitKey(fps_delay_ms)
        if key & 0xFF == ord("q"):
            # Exit if q pressed
            cv2.destroyAllWindows()
            break

        img = tello.read()
        if img is not None:
            img = tello.read()
            # The image received from tello is RGB, OpenCV works in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Show the image
            cv2.imshow("tello", img)

            # Save the images
            if args.save_img:
                cv2.imwrite(f"{str(save_dir)}/{save_frame_count:07d}.png", img)
                save_frame_count += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--local_ip", metavar="local_ip", type=str, required=True)
    parser.add_argument("--local_port", metavar="local_port", type=int, required=True)
    parser.add_argument("--save_img", metavar="save_img", type=bool, default=False)
    parser.add_argument("--save_dir", metavar="save_dir", type=str, default="dataset")
    parser.add_argument("--fps", metavar="fps", type=int, default=30)

    args, _ = parser.parse_known_args()
    main(args)