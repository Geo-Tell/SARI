Path='./res/SPSNet_deepglobe'
Root='xx/DeepGlobe/data_test'
Backbone='Resnet50'
pth='best.pth'
ModelFile='models.SPSNet_deepglobe_feat'

CUDA_VISIBLE_DEVICES=0,1 python -u evaluate.py \
--Root ${Root} \
--Path ${Path} \
--PthFile ${pth} \
--Backbone ${Backbone} \
--ModelFile ${ModelFile} \
--pngPath test/best_TTA \
--Validset test_CVPR.lst \
--dataset deepglobe \
--Background \
--Crop \
--Savepng \
--OverlapWeight \
--TTA
