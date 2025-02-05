import copy
import os
import torch
import logging

from tqdm import tqdm

from diveslowlearnfast.config import parse_args, save_config, to_dict, load_config
from diveslowlearnfast.eval.run_eval_epoch import run_eval_epoch
from diveslowlearnfast.models import SlowFast, save_checkpoint, load_checkpoint, get_parameter_count, \
    load_checkpoint_compat
from diveslowlearnfast.models.utils import last_checkpoint
from diveslowlearnfast.train import run_train_epoch, run_warmup, save_stats, load_stats, run_test_epoch, \
    MultigridSchedule
from diveslowlearnfast.train import helper as train_helper
from diveslowlearnfast.train.stats import StatsDB

logger = logging.getLogger(__name__)


def set_log_level():
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))


def print_device_props(device):
    print(f'Running on {device}')

    if device == torch.device('cpu'): return

    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"Device: {device_props.name}")
    print(f"Total memory: {device_props.total_memory / 1024 ** 2:.2f} MB")
    print(f"GPU number: {device_props.major}.{device_props.minor}")


def main():
    set_log_level()
    args = parse_args()
    is_eval_enabled = args.EVAL.ENABLED
    is_train_enabled = args.TRAIN.ENABLED
    eval_result_dir = args.EVAL.RESULT_DIR
    checkpoint_filename = args.TRAIN.CHECKPOINT_FILENAME
    config_path = os.path.join(args.TRAIN.RESULT_DIR, 'config.json')

    if os.path.exists(config_path):
        logger.info(f'Loading config from {config_path}, arguments are ignored')
        cfg = load_config(config_path)
        cfg.DATA.DATASET_PATH = args.DATA.DATASET_PATH
    else:
        logger.info(f'Saving config to {config_path}')
        cfg = args
        dict_cfg = to_dict(copy.deepcopy(cfg))
        save_config(dict_cfg, config_path)

    # always override these settings, so we can reuse the config that was saved on disk but have different
    # behaviour depending on whether one of these was enabled
    cfg.EVAL.ENABLED = is_eval_enabled
    cfg.EVAL.RESULT_DIR = eval_result_dir
    cfg.TRAIN.ENABLED = is_train_enabled
    if checkpoint_filename and len(checkpoint_filename) > 0:
        cfg.TRAIN.CHECKPOINT_FILENAME = checkpoint_filename

    dict_cfg = to_dict(copy.deepcopy(cfg))
    print(dict_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_props(device)

    # get the test objects before applying the multigrid config, this ensures the test batch size won't change
    labels = train_helper.get_include_labels(cfg)
    test_loader = train_helper.get_test_objects(cfg, labels)
    multigrid_schedule = None
    if cfg.MULTIGRID.SHORT_CYCLE or cfg.MULTIGRID.LONG_CYCLE:
        multigrid_schedule = MultigridSchedule()
        cfg = multigrid_schedule.init_multigrid(cfg)

    model = SlowFast(cfg).to(device)
    criterion, optimiser, train_loader, train_dataset, scaler = train_helper.get_train_objects(cfg, model)

    if cfg.DATA.THRESHOLD > 0 and train_dataset.num_classes != cfg.MODEL.NUM_CLASSES:
        logger.info(
            f'Threshold set and actual num_classes is less than specified in the config, updating to: {train_dataset.num_classes}.')
        cfg.MODEL.NUM_CLASSES = train_dataset.num_classes

    # If there is a checkpoint_path it is not worth using these weights
    if len(cfg.TRAIN.WEIGHTS_PATH) > 0:
        logger.info(f'Using pre-trained weights from {cfg.TRAIN.WEIGHTS_PATH}')
        load_checkpoint_compat(cfg.TRAIN.WEIGHTS_PATH, model, convert_from_caffe2=True, data_parallel=False)
        model = model.to(device)

    start_epoch = 1
    checkpoint_path = last_checkpoint(cfg)
    if cfg.TRAIN.AUTO_RESUME and checkpoint_path is not None:
        model, optimiser, epoch = load_checkpoint(
            model,
            optimiser,
            checkpoint_path,
            device
        )
        start_epoch = epoch + 1

    print(f'Start training model:')
    parameter_count = get_parameter_count(model)
    print(f'parameter count = {parameter_count}')
    # 4bytes x 3 (1model + 1gradient + 1optimiser)
    total_parameter_bytes = parameter_count * 12
    print(f'model size      = {total_parameter_bytes / 1024 ** 2:.3f} MB')
    print(f'from checkpoint = {checkpoint_path}')

    stats_db = StatsDB(cfg.TRAIN.STATS_DB)
    if cfg.EVAL.ENABLED and checkpoint_path:
        eval_stats = {}
        logger.info('Running eval epoch')
        stats = run_eval_epoch(model, test_loader, device, cfg, train_dataset.labels, eval_stats, stats_db, scaler)
        print(f'Eval epoch complete, saving stats to {cfg.TRAIN.RESULT_DIR}')
        save_stats(stats, cfg.EVAL.RESULT_DIR)
        return

    if not cfg.TRAIN.ENABLED:
        return

    if cfg.MODEL.COMPILE:
        gpu_ok = False
        if torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                gpu_ok = True

        if not gpu_ok:
            logger.warning(
                'GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower '
                'than expected.'
            )
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model, mode='reduce-overhead')

    model.train()
    if checkpoint_path is None:
        run_warmup(model, optimiser, criterion, train_loader, device, cfg)
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    stats = load_stats(os.path.join(cfg.TRAIN.RESULT_DIR, 'stats.json'))
    epoch_bar = tqdm(range(start_epoch, cfg.SOLVER.MAX_EPOCH), desc=f'Train epoch')
    for epoch in epoch_bar:
        # if the threshold is set and the seed is None we want to reload the dataset and loader at each epoch
        # this will ensure a new sample from the dataset with the threshold is drawn so the model will see a higher
        # variety of data.
        if cfg.DATA.THRESHOLD != -1 and cfg.DATA.SEED == -1:
            logger.info(f'Threshold = {cfg.DATA.THRESHOLD} and cfg.DATA.SEED == -1, reloading dataset and loader at epoch {epoch}')
            train_loader, train_dataset = train_helper.get_train_loader_and_dataset(cfg)

        if multigrid_schedule:
            cfg, changed = multigrid_schedule.update_long_cycle(cfg, epoch)
            model = SlowFast(cfg).to(device)
            criterion, optimiser, train_loader, train_dataset, scaler = train_helper.get_train_objects(cfg, model)
            multigrid_schedule.set_dataset(train_dataset, cfg)
            checkpoint_path = last_checkpoint(cfg)
            if checkpoint_path:
                model, optimiser, _ = load_checkpoint(
                    model,
                    optimiser,
                    checkpoint_path,
                    device
                )
            logger.info(
                f'multigrid long cycle shape=({cfg.TRAIN.BATCH_SIZE}x{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TRAIN_CROP_SIZE})')

        train_acc, train_loss = run_train_epoch(
            model,
            criterion,
            optimiser,
            train_loader,
            device,
            cfg,
            stats_db,
            epoch,
            multigrid_schedule,
            scaler
        )
        epoch_bar.set_postfix({ 'acc': f'{train_acc:.3f}', 'train_loss': f'{train_loss:.3f}' })

        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(model, optimiser, epoch, cfg)

        stats['train_losses'].append(float(train_loss))
        stats['train_accuracies'].append(float(train_acc))

        if epoch % cfg.TRAIN.EVAL_PERIOD == 0:
            model.eval()
            test_acc = run_test_epoch(
                model,
                test_loader,
                device,
                cfg,
                stats_db,
                epoch,
                scaler,
            )
            if len(stats['test_accuracies']) == 0:
                save_checkpoint(model, optimiser, epoch, cfg, 'best.pth')
            elif test_acc > stats['test_accuracies'][-1]: # the current model is better than the previous model
                save_checkpoint(model, optimiser, epoch, cfg, 'best.pth')

            model.train()
            stats['test_accuracies'].append(float(test_acc))

        save_stats(stats, cfg.TRAIN.RESULT_DIR)

        print('\n')
        print(10 * '_')


if __name__ == '__main__':
    main()
