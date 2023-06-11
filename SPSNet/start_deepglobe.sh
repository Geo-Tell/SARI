Path='./res/SPSNet_deepglobe'
Root='xx/DeepGlobe/data_768'
Backbone='Resnet50'
ModelFile='models.SPSNet_deepglobe_feat'
Port='12345'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port ${Port} train_sp.py \
--resPath ${Path} \
--Root ${Root} \
--Backbone ${Backbone} \
--dataset deepglobe \
--Trainset train.lst \
--Validset val.lst \
--Epoch 125 \
--Delay 75 \
--HFTrans \
--VFTrans \
--RRTrans \
--lrStart1 3e-4 \
--lrStart2 3e-4 \
--lamda1 0.1 \
--lamda2 0.1 \
--lamda3 50 \
--ModelFile ${ModelFile} \
--BestNum 1 \
--ClassWeight \
--Background \
# --Resume
