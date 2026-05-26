_base_ = '../../../mmaction2/configs/_base_/default_runtime.py'

NUM_CLASSES = 67

#mediapipe의 skeleton에 대한 정보
mediapipe_sign_layout = dict(
    num_node=65, #pose 23 + left hand 21 + right hand 21
    #joint간의 연결관계 (child, parent)
    #pose graph는 단순하게 
    inward=[ 
        # -------------------------
        # Pose 0~22
        # -------------------------
        #face
        (2, 0), (7, 2),
        (5, 0), (8, 5),
        (9, 0), (10, 0),
        # shoulders / arms
        (11, 0), (12, 0),
        (13, 11), (14, 12),
        (15, 13), (16, 14),

        # pose hand-related points
        (17, 15), (19, 15), (21, 15),
        (18, 16), (20, 16), (22, 16),

        # -------------------------
        # Left hand 23~43
        # -------------------------
        (23, 16),  # hand wrist(mirrored) to pose wrist(not mirrored)

        (24, 23), (25, 24), (26, 25), (27, 26),  # thumb
        (28, 23), (29, 28), (30, 29), (31, 30),  # index
        (32, 23), (33, 32), (34, 33), (35, 34),  # middle
        (36, 23), (37, 36), (38, 37), (39, 38),  # ring
        (40, 23), (41, 40), (42, 41), (43, 42),  # pinky

        # palm links
        (28, 24), (32, 28), (36, 32), (40, 36),

        # -------------------------
        # Right hand 44~64
        # -------------------------
        (44, 15),  # hand wrist(mirrored) to pose wrist(not mirrored)

        (45, 44), (46, 45), (47, 46), (48, 47),  # thumb
        (49, 44), (50, 49), (51, 50), (52, 51),  # index
        (53, 44), (54, 53), (55, 54), (56, 55),  # middle
        (57, 44), (58, 57), (59, 58), (60, 59),  # ring
        (61, 44), (62, 61), (63, 62), (64, 63),  # pinky

        # palm links
        (49, 45), (53, 49), (57, 53), (61, 57),
    ],
    #skeleton graph의 기준점 역할을 하는 노드
    center=0 #nose
)

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        in_channels=4, #x, y, z (depth), score
        graph_cfg=dict(
            layout=mediapipe_sign_layout,
            mode='stgcn_spatial'
        )
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=NUM_CLASSES,
        in_channels=256
    )
)

dataset_type = 'PoseDataset'
ann_file = '../dataset/gloss_sequences_splited/mediapipe_sign_3d.pkl'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mediapipe-sign-3d',
            name='stgcn_8xb16-joint-u100-80e_mediapipe-sign-keypoint-3d',
            config=dict(num_classes=NUM_CLASSES)),
        define_metric_cfg=[
            dict(name='epoch'),
            dict(name='step'),
            dict(name='lr', step_metric='epoch'),
            dict(name='loss', step_metric='epoch'),
            dict(name='loss_cls', step_metric='epoch'),
            dict(name='top1_acc', step_metric='epoch'),
            dict(name='top5_acc', step_metric='epoch'),
            dict(name='acc/*', step_metric='epoch'),
        ])
]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# j	 	관절 좌표 자체
# b		연결된 관절 간 차이 벡터
# jm	시간에 따른 joint 변화량
# bm	시간에 따른 bone 변화량

train_pipeline = [
    # 3D면 PreNormalize3D를 쓸 수 있지만,
    # MediaPipe z scale이 특수하므로 처음에는 직접 전처리하고 생략하는 것도 가능.
    # dict(type='PreNormalize3D'),

    dict(type='GenSkeFeat', feats=['j']), #모델 입력용 skeleton feature (검출 X, formatting)
    dict(type='UniformSampleFrames', clip_len=100), #프레임 수 통일
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')))
val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='test',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1) #20
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=30, #16
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(checkpoint=dict(interval=10), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
