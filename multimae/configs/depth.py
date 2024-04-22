import ml_collections


def depth_configs():
    config = ml_collections.ConfigDict()

    config.run_name = "debug-only-semseg"
    config.log_dir = "./logs/"
    config.cons_lvl = "INFO"
    config.file_lvl = "INFO"
    config.save_every_epoch = False

    config.device = "cuda"

    config.in_domains = ["rgb", "semseg"]
    config.out_domains = ["depth"]
    config.decoder_main_tasks = ["rgb", "semseg"]

    config.semseg_num_classes = 256

    config.patch_size = 16
    config.input_size = 224

    config.fine_tune_path = "../../data/dq/mae-b_dec512d8b_1600e_multivit-c477195b.pth"

    config.lr = 3e-4
    config.weight_decay = 1e-4
    config.total_epochs = 1
    config.batch_size = 16

    config.train_dir = "../../data/dq/clevr_complex/train"
    config.val_dir = "../../data/dq/clevr_complex/val"

    config.seed = 0xAB0BA
    return config

def embedding_configs():
    config = ml_collections.ConfigDict()

    config.device = "cuda"

    config.in_domains = ["rgb", "semseg"]
    config.out_domains = ["depth"]
    config.decoder_main_tasks = ["rgb", "semseg"]

    config.semseg_num_classes = 256

    config.patch_size = 16
    config.input_size = 224

    config.fine_tune_path = "/home/MultiModalCoreset/fastapi/ckpt/multi.ckpt"

    config.batch_size = 64

    config.train_dir = "/home/data/dq/clevr_complex/val"
    config.embed_save_path = '/home/data/dq/clevr_complex/multimae_embeds.npy'

    config.seed = 0xAB0BA
    return config
