# tello_drawer

Draw a shape in space with your hand and make the drone replicate this shape!
The drone stays in the air and watches your hand by the camera.
The images from the camera are being sent to the PC, where your hand and its pose are detected.
The detected hand movement is then converted to drone steering commands which makes the drone replicate your movement.


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

Create 'models' directory
```
cd src
mkdir models
```
Add model config and weights into 'models' directory

## Using the repository
### Dataset saver
```
python3 dataset_saver.py --local_ip "0.0.0.0" --local_port 8889 --save_img True 
```
- Set fps with `--fps` flag
- Set dataset saving directory with `--save_dir`
