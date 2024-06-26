_base_ = ["icod_train.py"]

num_frames = 5

__BATCHSIZE = 2

train = dict(
    batch_size=__BATCHSIZE,
    use_amp=True,
    num_epochs=10,
    lr=0.0001,
    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=0),
        mode="constant",
        cfg=dict(coef=1),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),
    data=dict(
        shape=dict(h=384, w=384),
        names=["moca_mask_tr"],
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    data=dict(
        shape=dict(h=384, w=384),
        names=["cad", "moca_mask_te"],
    ),
)
