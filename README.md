
# Installation

```
git clone git@github.com:veesion-io/MogFace.git
cd MogFace
nvidia-docker run --gpus all --name blurring --security-opt seccomp=unconfined \
  --net=host --ipc=host -v /dev/shm:/dev/shm --ulimit memlock=-1 \
  -v /path/to/MogFace:/workspace/ -v /path/to/your/videos:/workspace/videos/ \
  --ulimit stack=67108864 -it nvcr.io/nvidia/pytorch:23.06-py3 /bin/bash

cd /workspace/
pip install gdown
gdown https://drive.google.com/uc?id=1s8LFXQ5-zsSRJKVHLFqmhow8cBn4JDCC
mkdir -p snapshots/MogFace && mv model_140000.pth snapshots/MogFace/
mkdir annotations

cd utils/nms && python setup.py build_ext --inplace && cd ../..
cd utils/bbox && python setup.py build_ext --inplace && cd ../..

pip uninstall -y opencv-contrib-python \
  && rm -rf /usr/local/lib/python3.10/dist-packages/cv2
pip install opencv-contrib-python ffprobe3

apt update && DEBIAN_FRONTEND=noninteractive apt install ffmpeg -y
```

# Detect and blur faces : 

Detect faces with a deep learning model and save .txt annotations files.
```
CUDA_VISIBLE_DEVICES=0 python test_multi.py -c configs/mogface/MogFace.yml -n 140
```

Use the created files to create videos with faces blurred : 
```
python3 blur_detected_faces.py
```

# Optimize the model

Download the calibration data :
```
gdown https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q
unzip WIDER_val.zip
python3 preprocess_calibration_images.py
find WIDER_val/resized_images -type f -name "*.jpg" > calibration_file.txt
pip install pycuda
apt update && apt install libgl-dev -y
```

Optimize the model : 
```
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx     --fp16 --best --workspace=9000 --saveEngine=model_fp16.trt     --inputIOFormats=fp32:chw --allowGPUFallback --outputIOFormats=fp32:chw --calib=calibration_file.txt
```