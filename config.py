n_classes = 35    # total number of object classes
rgb_channels = n_features = 3
train_dir = [
    "./data/cityscrapes/gtFine_trainvaltest/gtFine/train",
    "./data/cityscrapes/leftImg8bit_trainvaltest/leftImg8bit/train"
]
phase1_dir = "./checkpoints/phase1_model.pth"
phase2_dir = "./checkpoints/phase2_model.pth"

epochs = 200         # total number of train epochs
decay_after = 100    # number of epochs with constant lr
lr = 0.0002
betas = (0.5, 0.999)