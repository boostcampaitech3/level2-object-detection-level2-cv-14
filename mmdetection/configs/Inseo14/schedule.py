# optimizer
optimizer_config = dict(grad_clip=None)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
            }
        )
    )

# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1465,
    warmup_ratio=0.001,
    periods=[5860, 5860, 7325, 7325, 8790],
    restart_weights=[0.75, 0.7, 0.65, 0.6, 0.55],
    by_epoch=False,
    min_lr=5e-6
    )
runner = dict(type='EpochBasedRunner', max_epochs=36)
