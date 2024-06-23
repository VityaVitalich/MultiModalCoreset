import ml_collections


def depth_configs():
    config = ml_collections.ConfigDict()

    config.run_name = "rgb-semseg-40epoch_coreset_sum_20"
    config.log_dir = "./logs/"
    config.cons_lvl = "INFO"
    config.file_lvl = "INFO"
    config.save_every_epoch = True

    config.device = "cuda"

    config.in_domains = ["semseg", "rgb"]
    config.out_domains = ["depth"]
    config.decoder_main_tasks = ["rgb", "semseg"]

    config.semseg_num_classes = 256

    config.patch_size = 16
    config.input_size = 224

    config.fine_tune_path = "/home/cache/data/dq/multimae.pth"

    config.lr = 5e-5
    config.weight_decay = 0
    config.total_epochs = 40
    config.batch_size = 128

    config.train_dir = "/home/cache/data/dq/clevr/train"
    config.subset_idx = "/home/cache/data/dq/clevr/train/multimae_sum_index_20.pickle"
    config.val_dir = "/home/cache/data/dq/clevr/val"

    config.seed = 0xAB0BA
    return config

def embedding_configs():
    config = ml_collections.ConfigDict()

    config.aggregation = 'sum'
    config.device = "cuda"

    config.in_domains = ["rgb", "semseg"]
    config.out_domains = ["depth"]
    config.decoder_main_tasks = ["rgb", "semseg"]

    config.semseg_num_classes = 256

    config.patch_size = 16
    config.input_size = 224

    config.fine_tune_path = "/home/MultiModalCoreset/multimae/ckpt/rgb-semseg-40epoch_2024-04-27_21:21:34/epoch__0036_-_rmse__0.003263.ckpt"

    config.batch_size = 1024

    config.train_dir = "/home/cache/data/dq/clevr/train"
    config.embed_save_path = '/home/cache/data/dq/clevr/train/multimae_sum_embeds.npy'

    config.seed = 0xAB0BA
    return config
