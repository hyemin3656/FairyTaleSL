# Custom TSN config for MMAction2
_base_ = [
    '../../mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'  # inherit template config
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=67))  # change from 400 to 67


# dataset settings
dataset_type = 'VideoDataset'
data_root_train = 'data/ksl_video/ksl_video_train'
data_root_val = 'data/ksl_video/ksl_video_val'
data_root_test = 'data/ksl_video/ksl_video_test'
ann_file_train = 'data/ksl_video/train_list.txt'
ann_file_val = 'data/ksl_video/val_list.txt'
ann_file_test = 'data/ksl_video/test_list.txt'


train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root_train)))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test)))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.001, # change from 0.01 to 0.001
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

model = dict(
    cls_head=dict(
        dropout_ratio=0.7))
# use the pre-trained model for the whole TSN network. model path can be found in model zoo
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
