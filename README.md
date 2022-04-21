# One Team

![Untitled](https://user-images.githubusercontent.com/64190071/164357692-6ab59eb0-d522-495b-ba49-60221bd4f6a6.png)

- 2022.03.21 ~ 2022.04.07
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- 재활용 품목 분류를 위한 Object Detection
- Public & Private 1st prize! 🏆

## MEMBERS

|                                                  [김찬혁](https://github.com/Chanhook)                                                   |                                                                          [김태하](https://github.com/TaehaKim-Kor)                                                                           |                                                 [문태진](https://github.com/moontaijin)                                                  |                                                                        [이인서](https://github.com/Devlee247)                                                                         |                                                                         [장상원](https://github.com/agwmon)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![KakaoTalk_20220421_103825891_03](https://user-images.githubusercontent.com/64190071/164358205-2048f3c2-1216-4836-a77f-a25de6a9091c.jpg) | ![KakaoTalk_20220421_103825891](https://user-images.githubusercontent.com/64190071/164358113-c8db12e4-15d1-469c-8026-cd0a5cb89e36.jpg) | ![KakaoTalk_20220421_103825891_04](https://user-images.githubusercontent.com/64190071/164358227-ef0d7919-bd0d-4a9d-8d50-42757a5c3534.jpg) | ![KakaoTalk_20220421_103825891_02](https://user-images.githubusercontent.com/64190071/164358185-a63371d7-84ad-4eb9-8337-c70857c0e170.jpg) | ![KakaoTalk_20220421_103825891_01](https://user-images.githubusercontent.com/64190071/164358129-a9ce91f8-84c5-4a9c-8329-27cf18e68e7f.jpg) |

## 문제 정의(대회소개) & Project Overview

![Untitled 2](https://user-images.githubusercontent.com/64190071/164357707-420bb60c-74f3-4aba-946f-a47dbc9edc24.png)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## Dataset

- 전체 이미지 개수 : 9754장
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)

## METRICS

- mAP50

![Untitled 3](https://user-images.githubusercontent.com/64190071/164357745-4d03deb3-6104-4706-a890-3d002a904067.png)

![Untitled 4](https://user-images.githubusercontent.com/64190071/164357754-718a8628-872e-4f1e-9d12-4e212b2444ab.png)

![Untitled 5](https://user-images.githubusercontent.com/64190071/164357763-9d7c667a-2c5a-4b92-b4ae-6c32be0b7d34.png)

### LB Score

Public, Private 1등!

![Untitled 6](https://user-images.githubusercontent.com/64190071/164357778-a4a08ae4-095c-48ca-bdd1-660badaead18.png)
![Untitled 7](https://user-images.githubusercontent.com/64190071/164357785-10bca7d7-84a7-483a-b688-e9ff837dbe72.png)

### TOOLS

- Github (Custom Git-flow Branching, Issue, PR)
- Notion
- Slack
- Wandb

## Project Outline

![Untitled 8](https://user-images.githubusercontent.com/64190071/164357814-9ab3babd-4860-4f8c-aeec-5110c8e96848.png)

- [x]  Basecode
- [x]  EDA
- [x]  Model Search
- [x]  Model Experiments
- [x]  Augmentation Experiments
- [x]  Pseudo Labeling
- [x]  Ensemble

## Experiments

![Untitled 9](https://user-images.githubusercontent.com/64190071/164357831-5d34f03e-cdf8-4483-b276-f95a44aaa8f5.png)

### Models

- Cascade RCNN
- Retinanet
- Vfnet
- TOOD
- Yolov5
- YoloX
- Centernet
- Atss
- DetectoRS
- DETR
- Dynamic RCNN
- Deformal DETR
- NAS FCOS
- Autoassign
- SparseRCNN

### Backbones

- Swin Transformer
- Resnet
- Res2net
- ResNEXT

### Schedulers

- Step
- Cosine Annealing Restart

### Augmentations

- Albumentation built-in augmentations
- Multi-Scale training
- Mosaic
- Mix-up
- TTA (Multi-scale Testing, Flip)

## Results

![Untitled 10](https://user-images.githubusercontent.com/64190071/164357839-7476d444-9827-4067-800a-b0bdea9eb147.png)


## INSTALLATION GUIDE

### MMDetection

```bash
# python 3.7버전의 openmmlab이라는 이름의 가상환경 생성
conda create -n openmmlab python==3.7 -y
conda activate openmmlab

# library 설치
conda install pytorch=1.7.0 torchvision cudatoolkit=11.0 -c pytorch
conda install pandas tqdm

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install mmdet

cd mmdetection
pip install -r requirements/albu.txt

# log dep
pip install wandb mlflow
```

## Yolov5

```bash
# python 3.8 버전의 가상환경 생성
conda create -n yolov5 python=3.8

# 가상환경 활성화
conda activate yolov5

cd yolov5
pip install -r requirements.txt

conda install psutil
pip install joblib wandb
```

## Citation

### MMDetection

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
