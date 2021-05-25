# tello_drawer

Draw a shape in space with your hand and make the drone replicate this shape!
The drone stays in the air and watches your hand by the camera.
The images from the camera are being sent to the PC, where your hand and its pose are detected.
The detected hand movement is then converted to drone steering commands which makes the drone replicate your movement.

**TODO**: Add images and information which gesture means what

## Prerequisites
- Python 3.8

There are some required packages to install on the PC. Run the following command to install them
```
sudo apt install libswscale-dev libavcodec-dev libavutil-dev python3-virtualenv
```

## Cloning the repository
To clone the repository use the following lines
```
git clone --recurse-submodules https://github.com/arekmula/tello_drawer
cd tello_drawer
```

Create and activate virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install requirements
```
pip install -r requirements.txt
cd h264decoder
pip install .
```

Create `models` directory
```
cd src
mkdir models
```

Download detector model and its weights along classification model from the **Releases** page and add it to the `models`
directory.

## Using the repository
### Tello Drawer
To run the Tello Drawer use following commands:
- To run with the the Tello drone:
```
python3 main.py --image_source "tello" --local_ip "0.0.0.0" --local_port 8889
```

- You can also run the test drawing with your PC built-in camera or video that you recorded earlier.
```
python3 main.py --image_source "built_camera" --camera_index 0
python3 main.py --image_source "saved_file" --filepath "path/to/file"
```


### Dataset saver
The dataset saver helps in gathering the data using the Tello drone for further processing.
It connects to the Tello drone, activates the video stream, and saves each received frame.
```
python3 dataset_saver.py --local_ip "0.0.0.0" --local_port 8889 --save_img True 
```
- Set fps with `--fps` flag
- Set dataset saving directory with `--save_dir`


### Hand detection
To detect hands on the image we utilized cansik's YOLO hand detector which is available
[here](https://github.com/cansik/yolo-hand-detection).
We haven't made any changes to the detector. 

### Hand classification
We have to split the hand detections into 2 separate classes.
The fist is responsible for the start/stop signal while the palm is responsible for drawing. To do so we created classificator: 
**TODO**