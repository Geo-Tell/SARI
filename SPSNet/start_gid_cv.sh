Path='./res/SPSNet_gid_cv'
Root='xx/GID/data_1024'
Backbone='Resnet101'
ModelFile='models.SPSNet_gid_feat'
Port='12345'
for ((i=0;i<5;i++));
do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port ${Port} train_sp_cv.py \
    --resPath ${Path} \
    --Root ${Root} \
    --Backbone ${Backbone} \
    --dataset gid \
    --Trainset trainval.lst \
    --Validset trainval.lst \
    --Epoch 225 \
    --Delay 75 \
    --NowFold ${i} \
    --Folds 5 \
    --HFTrans \
    --VFTrans \
    --RRTrans \
    --RCTrans \
    --lrStart1 3e-4 \
    --lrStart2 1e-4 \
    --lamda1 0.1 \
    --lamda2 0.1 \
    --lamda3 50 \
    --ModelFile ${ModelFile} \
    --BestNum 1 \
    ;
done
