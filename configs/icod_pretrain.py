_base_ = ["icod_train.py"]

has_test = False

__BATCHSIZE = 4
__NUM_EPOCHS = 150
__NUM_TR_SAMPLES = 3040
__ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE  # drop_last is True
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

train = dict(
    batch_size=__BATCHSIZE,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
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
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="step",
        cfg=dict(
            milestones=int(__NUM_ITERS * 2 / 3),
            gamma=0.1,
        ),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),
    data=dict(
        shape=dict(h=384, w=384),
        names=["cod10k_tr"],
    ),
)

test = dict(
    data=dict(
        shape=dict(h=384, w=384),
        names=[],
    ),
)
