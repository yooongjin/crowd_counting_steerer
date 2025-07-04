# Getting started 

## preparation 

- **Clone this repo** in the directory (```root/```):


```bash
cd $root
git clone https://github.com/yooongjin/crowd_counting_steerer.git
```
- **Install dependencies.** We use python 3.9 and pytorch >= 1.12.0 : http://pytorch.org.

```bash
conda create -n STEERER python=3.9 -y
conda activate STEERER
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
cd ${STEERER}
pip install -r requirements.txt
pip install decord
```

## Video Inference

```bash
python video_inference.py  --pretrained [PRETRAINED_MODEL] --sample_rate [SAMPLE_RATE] --output_path [OUTPUT_PATH] --batch_size [BATCH_SIZE] --video_path [VIDEO_PATH] 
```
