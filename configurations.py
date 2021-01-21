model = {
    "CNN" : {
        "mixup": False,
        "cutout": False,
        "data_augmentation": False,
        "early_stop": False,
    },

    "CNN_early": {
        "mixup": False,
        "cutout": False,
        "data_augmentation": False,
        "early_stop": True,
    },

    "CNN_augmented" : {
        "mixup": False,
        "cutout": False,
        "data_augmentation": True,
        "early_stop": False,
    },

    "CNN_mixup" : {
        "mixup": True,
        "cutout": False,
        "data_augmentation": True,
        "early_stop": False,
    },

    "CNN_cutout" : {
        "mixup": False,
        "cutout": True,
        "data_augmentation": True,
        "early_stop": False,
    },

    "CNN_both" : {
        "mixup": True,
        "cutout": True,
        "data_augmentation": True,
        "early_stop": False,
    },


    "ResNet" : {
        "mixup": False,
        "cutout": False,
        "data_augmentation": False,
        "early_stop": False,
    },
    "ResNet_augmented" : {
        "mixup": False,
        "cutout": False,
        "data_augmentation": True,
        "early_stop": False,
    },
    "ResNet_mixup" : {
        "mixup": True,
        "cutout": False,
        "data_augmentation": True,
        "early_stop": False,
    },
    "ResNet_cutout" : {
        "mixup": False,
        "cutout": True,
        "data_augmentation": True,
        "early_stop": False,
    },
    "ResNet_both" : {
        "mixup": True,
        "cutout": True,
        "data_augmentation": True,
        "early_stop": False,
    },




}
