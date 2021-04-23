# tello_drawer



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
