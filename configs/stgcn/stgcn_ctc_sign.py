_base_ = './stgcn_8xb16-joint-u100-80e_mediapipe-sign-keypoint-3d.py'

# Number of real gloss ids. Existing ids are assumed to be 0..NUM_GLOSS-1.
# STGCNCTCHead internally uses blank_idx = NUM_GLOSS.
NUM_GLOSS = 67

# GPU CTC loss backward is not deterministic in PyTorch, so this config must
# keep deterministic algorithms disabled even if tools/train.py is called with
# --deterministic. Set the seed only for ordinary run-to-run stability.
randomness = dict(seed=0, diff_rank_seed=False, deterministic=False)

# The base config already defines RecognizerGCN + STGCN backbone.
# Only the classification head is replaced with the CTC head.
model = dict(
    type='RecognizerGCN',
    cls_head=dict(
        type='STGCNCTCHead',
        num_classes=NUM_GLOSS,
        in_channels=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.1))


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mediapipe-sign-3d',
            name='stgcn_ctc_sign',
            config=dict(num_gloss=NUM_GLOSS, blank_idx=NUM_GLOSS)),
        define_metric_cfg=[
            dict(name='epoch'),
            dict(name='step'),
            dict(name='lr', step_metric='epoch'),
            dict(name='loss', step_metric='epoch', summary='min'),
            dict(name='loss_ctc', step_metric='epoch', summary='min'),
            dict(name='wer', step_metric='epoch', summary='min'),
            dict(name='substitutions', step_metric='epoch', summary='min'),
            dict(name='deletions', step_metric='epoch', summary='min'),
            dict(name='insertions', step_metric='epoch', summary='min'),
            dict(name='inference_time', step_metric='epoch', summary='min'),
            dict(name='inference_time_per_sample', step_metric='epoch',
                 summary='min'),
            dict(name='inference_fps', step_metric='epoch', summary='max'),
            dict(name='num_inference_samples', step_metric='epoch'),
        ])
]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# For CTC, do not put the blank id in gt_gloss/label sequences.
# Example annotation target: dict(label=[0, 12, 5]) or dict(gt_gloss=[0, 12, 5]).
# Avoid UniformSampleFrames for sentence-level sign recognition because it can
# destroy temporal alignment. Prefer preserving the whole sequence, or use a
# deterministic pad/crop policy. The current annotation file has max 660
# frames, so MAX_SEQ_LEN=672 preserves all frames while keeping a fixed batch
# shape. Increase it if you regenerate annotations with longer sequences.
MAX_SEQ_LEN = 672

train_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),
    # dict(type='UniformSampleFrames', clip_len=MAX_SEQ_LEN),  # Not CTC safe.
    dict(type='PadTo', length=MAX_SEQ_LEN, mode='zero'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackCTCInputs')
]

val_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='PadTo', length=MAX_SEQ_LEN, mode='zero'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackCTCInputs')
]

test_pipeline = val_pipeline

ann_path = '../dataset/gloss_sequences_splited/mediapipe_sign_3d.pkl'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='PoseDataset',
        ann_file=ann_path,
        pipeline=train_pipeline,
        split='train'))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline, ann_file=ann_path))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, ann_file=ann_path))



# WER is lower-is-better, so CheckpointHook cannot use save_best='auto'.
default_hooks = dict(
    checkpoint=dict(interval=-1, save_best='wer', rule='less', save_last=True),
    logger=dict(interval=100, ignore_last=False))

# The gloss-level classifier used SGD lr=0.1, but CTC + a randomly initialized
# BiLSTM head is much easier to destabilize. Use a smaller LR and gradient
# clipping to avoid NaN/Inf CTC loss from exploding recurrent gradients.
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2))

# Load a gloss-level ST-GCN checkpoint after removing cls_head.* keys:
# python tools/convert_stgcn_backbone_ckpt.py gloss_level.pth backbone_only.pth
# load_from = 'backbone_only.pth'

# WER is computed at gloss-id sequence level. Lower is better.
val_evaluator = dict(_delete_=True, type='WERMetric')
test_evaluator = dict(_delete_=True, type='WERMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=5, val_interval=5) #20
val_cfg = dict(type='TimedValLoop')
test_cfg = dict(type='TimedTestLoop')

