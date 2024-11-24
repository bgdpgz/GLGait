# GLGait: A Global Local Temporal Receptive Field Network for Gait Recognition in the Wild
[Paper](https://arxiv.org/abs/2408.06834) has been accepted in ACM MM 2024. This is the code for it.
# Operating Environments
## Pytorch Environment
* Pytorch=1.11.0
# CheckPoints
* The checkpoint for Gait3D [link](https://pan.baidu.com/s/1quNAQ1pTOHUa3tpfGCQ7IQ?pwd=fue3).
* The checkpoint for GREW [link](https://pan.baidu.com/s/1H41p_FQjSkL8Jn_2xWWsLA?pwd=soci).
# Train and Test
## Train
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/GLGait/GLGait_Gait3D.yaml --phase train
```
## Test
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/GLGait/GLGait_Gait3D.yaml --phase test
```
* python -m torch.distributed.launch: DDP launch instruction.
* --nproc_per_node: The number of gpus to use, and it must equal the length of CUDA_VISIBLE_DEVICES.
* --cfgs: The path to config file.
* --phase: Specified as train or test.
# Acknowledge
The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait).
# Citation
```
@inproceedings{peng2024glgait,
  title={GLGait: A Global-Local Temporal Receptive Field Network for Gait Recognition in the Wild},
  author={Peng, Guozhen and Wang, Yunhong and Zhao, Yuwei and Zhang, Shaoxiong and Li, Annan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={826--835},
  year={2024}
}
```
