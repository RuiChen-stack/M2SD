_base_ = [
    '../_base_/models/m2sd.py', '../_base_/datasets/msd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='M2SD',
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    # neck=dict(type='Myneck', reverse=True),
    neck=dict(type='MSPM', reverse=True),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=3e-05, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=6000,
                 warmup_ratio=2.5e-7,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
evaluation = dict(  # 构建评估钩 (evaluation hook) 的配置文件。细节请参考 mmseg/core/evaluation/eval_hook.py。
    interval=1000,  # 评估的间歇点
    metric='mIoU')
checkpoint_config = dict(  # 设置检查点钩子 (checkpoint hook) 的配置文件。执行时请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py。
    by_epoch=False,  # 是否按照每个 epoch 去算 runner。
    interval=1000)  # 保存的间隔

