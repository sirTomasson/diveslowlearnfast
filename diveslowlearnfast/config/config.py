from diveslowlearnfast.config.defaults import Config


def merge_config(cfg: Config, args) -> Config:
    if args.data.annotations_path:
        cfg.DATA.ANNOTATIONS_PATH = args.data.annotations_path

    if args.data.vocab_path:
        cfg.DATA.VOCAB_PATH = args.data.vocab_path

    if args.data.videos_path:
        cfg.DATA.VIDEOS_PATH = args.data.videos_path
        
    if args.data_loader.use_decord:
        cfg.DATA_LOADER.USE_DECORD = args.data_loader.use_decord

    if args.data_loader.num_workers:
        cfg.DATA_LOADER.NUM_WORKERS = args.data_loader.num_workers

    return cfg