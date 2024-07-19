# GLGait: A Global Local Temporal Receptive Field Network for Gait Recognition in the Wild
Paper has been accepted in ACM MM 2024. This is the code for it.
# Operating Environments
## Pytorch Environment
* Pytorch=1.11.0
# CheckPoints
* The checkpoint for Gait3D [link](https://pan.baidu.com/s/1PCGMTZerjFQFl_eammv4MQ?pwd=1357).
* The checkpoint for GREW [link].
# Train and Test
## Train
Using the command: CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/GLGait/GLGait_Gait3D.yaml --phase train
## Test
Using the command: CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/GLGait/GLGait_Gait3D.yaml --phase test
* python -m torch.distributed.launch: DDP launch instruction.
* --nproc_per_node: The number of gpus to use, and it must equal the length of CUDA_VISIBLE_DEVICES.
* --cfgs: The path to config file.
* --phase: Specified as train or test.
