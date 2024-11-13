
defaults = {
    'SLOWFAST': {
        'ALPHA': 4,
        'BETA_INV': 8,
        'FUSION_CONV_CHANNEL_RATIO': 2,
        'FUSION_KERNEL_SZ': 7
    },
    'RESNET': {
      'ZERO_INIT_FINAL_BN': True,
      'WIDTH_PER_GROUP': 64,
      'NUM_GROUPS': 1,
      'DEPTH': 50,
      'TRANS_FUNC': 'bottleneck_transform',
      'STRIDE_1X1': False,
      'NUM_BLOCK_TEMP_KERNEL': [[3, 3], [4, 4], [6, 6], [3, 3]],
      'SPATIAL_STRIDES': [[1, 1], [2, 2], [2, 2], [2, 2]],
      'SPATIAL_DILATIONS': [[1, 1], [1, 1], [1, 1], [1, 1]],
    },
    'NONLOCAL': {
      'LOCATION': [[[], []], [[], []], [[], []], [[], []]],
      'GROUP': [[1, 1], [1, 1], [1, 1], [1, 1]],
      'INSTANTIATION': 'dot_product'
    },
    'BN': {
      'USE_PRECISE_STATS': True,
      'NUM_BATCHES_PRECISE': 200
    },
    'MODEL': {
      'NUM_CLASSES': 48,
      'ARCH': 'slowfast',
      'MODEL_NAME': 'SlowFast',
      'LOSS_FUNC': 'cross_entropy',
      'DROPOUT_RATE': 0.5,
    }
}