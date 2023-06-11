Path='./res/SPSNet_gid_cv'
Root='xx/GID/data_1024'
Backbone='Resnet101'
pth='best'
ModelFile='models.SPSNet_gid_feat'

CUDA_VISIBLE_DEVICES=0,1 python -u evaluate_cv.py \
--Root ${Root} \
--Path ${Path} \
--PthFile ${pth} \
--Backbone ${Backbone} \
--ModelFile ${ModelFile} \
--Validset trainval.lst \
--dataset gid \
--Crop \
--MultiScale
