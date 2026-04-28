import numpy as np
from mmengine.dataset import Compose


def build_frame_pipeline():
    return Compose([
        dict(type='Resize', scale=(-1, 256)),
        dict(type='TenCrop', crop_size=224),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='PackActionInputs')
    ])


def build_frame_data(frames):
    return dict(
        imgs=frames,
        frame_inds=np.arange(len(frames), dtype=np.int64),
        total_frames=len(frames),
        num_clips=len(frames),
        clip_len=1,
        modality='RGB',
        start_index=0
    )