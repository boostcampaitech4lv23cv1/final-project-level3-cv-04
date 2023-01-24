# final-project-level3-cv-04

ENVIROMENT SETTING  
```
conda create -n torchkpop python=3.7
conda activate torchkpop
conda install cudatoolkit=11 -c conda-forge
conda install cudatoolkit-dev=11 -c conda-forge 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
cd ..
pip install insightface
pip install pytube
pip install onnxruntime
pip install ffmpeg-python
pip install neptune-client
pip install wandb
pip install clearml
pip3 install tabulate
pip install yacs
pip install torchmetrics
pip install monai
python3 main.py
```