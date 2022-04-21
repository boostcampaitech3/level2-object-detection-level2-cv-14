_base_ = [
    'dataset.py',
    'schedule.py',
    'runtime.py',
    'models.py'
]

# 총 epochs 사이즈
runner = dict(max_epochs=50)

# samples_per_gpu -> batch size라 생각하면 됨
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2)

checkpoint_config = dict(interval=-1)

# 로그와 관련된 셋팅
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
                init_kwargs=dict(
                    # 각각 자신에 맞춰서 Project이름 설정
                    project= 'Sangwon_Jang',
                    name = 'atss_swin_l_1024_hard_fold0_mix&mosaic_prob'
                ),
            ),
        dict(type='MlflowLoggerHook')
    ])

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536])
)

find_unused_parameters = True