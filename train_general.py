import os
import argparse
import json
# import comet_ml
import asteroid.data.librimix_dataset

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import asteroid
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.engine.schedulers import DPTNetScheduler
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, SingleSrcNegSTOI
from asteroid.losses import stoi
from src.data import make_dataloaders
from src.engine.system import GeneralSystem
from src.losses.multi_task_wrapper import MultiTaskLossWrapper
from src.models import *
pl.seed_everything(42)

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="LibriMix", choices=["LibriMix", "wsj0-mix"])
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet"])
parser.add_argument("--strategy", default="multi_task", choices=["from_scratch", "pretrained", "multi_task"])
parser.add_argument("--exp_dir", default="exp/multi_task", help="Full path to save best validation model")
parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Total batch size = batch_size * accumulate_grad_batches")
parser.add_argument("--comet", action="store_true", help="Comet logger")
parser.add_argument("--resume", action="store_true", help="Resume-training")

known_args = parser.parse_known_args()[0]
if known_args.strategy == "pretrained":
    parser.add_argument("--load_path", default=r"D:\Project\SSL-pretraining-separation-main\model\ConvTasNet\best_model.pth", help="Checkpoint path to load for fine-tuning.")
elif known_args.strategy == "multi_task":
    parser.add_argument("--train_enh_dir", default="D:/Project/SSL-pretraining-separation-main/data/librimix/Libri2Mix/wav16k/min/metadata/train-100", help="Multi-task data dir.")
    parser.add_argument("--train_enh_and_sep_dir",default="D:/Project/SSL-pretraining-separation-main/data/librimix/Libri2Mix/wav16k/min/metadata/train-100",
                        help="Multi-task data dir.")

if known_args.resume:
    parser.add_argument("--resume_ckpt", default="last.ckpt", help="Checkpoint path to load for resume-training")
    if known_args.comet:
        parser.add_argument("--comet_exp_key", default=None, required=True, help="Comet experiment key")


def main(conf):
    train_enh_dir = conf["main_args"].get("train_enh_dir", None)
    train_enh_and_sep_dir = conf["main_args"].get("train_enh_and_sep_dir", None)
    resume_ckpt = conf["main_args"].get("resume_ckpt", None)

    train_loader, val_loader, train_set_infos = make_dataloaders(
        corpus=conf["main_args"]["corpus"],
        train_dir=conf["data"]["train_dir"],
        val_dir=conf["data"]["valid_dir"],
        train_enh_dir=train_enh_dir,
        train_enh_and_sep_dir = train_enh_and_sep_dir,
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
    )

    conf["masknet"].update({"n_src": conf["data"]["n_src"]})
    if conf["main_args"]["strategy"] == "multi_task":
        conf["masknet"].update({"n_src": conf["data"]["n_src"]+3})
    print("masknet num:",conf["masknet"]["n_src"])
    if conf["main_args"]["strategy"] == "pretrained":
        model = getattr(asteroid.models, conf["main_args"]["model"]).from_pretrained(conf["load_path"])
    else:

        model = getattr(asteroid.models, conf["main_args"]["model"])(**conf["filterbank"], **conf["masknet"])

    optimizer = make_optimizer(model.parameters(), **conf["optim"])

    # Define scheduler
    scheduler = None
    if conf["main_args"]["model"] in ["DPTNet", "SepFormerTasNet", "SepFormer2TasNet"]:
        steps_per_epoch = len(train_loader) // conf["main_args"]["accumulate_grad_batches"]
        conf["scheduler"]["steps_per_epoch"] = steps_per_epoch
        scheduler = {
            "scheduler": DPTNetScheduler(
                optimizer=optimizer,
                steps_per_epoch=steps_per_epoch,
                d_model=model.masker.mha_in_dim,
            ),
            "interval": "batch",
        }
    elif conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    pit_wrapper = MultiTaskLossWrapper if conf["main_args"]["strategy"] == "multi_task" else PITLossWrapper
    loss_func = pit_wrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = GeneralSystem(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, filename='{epoch}-{step}', monitor="val_loss", mode="min",
        save_top_k=conf["training"]["epochs"], save_last=True, verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    loggers = []
    tb_logger = pl.loggers.TensorBoardLogger(
        os.path.join(exp_dir, "tb_logs/"),
    )
    loggers.append(tb_logger)
    if conf["main_args"]["comet"]:
        comet_logger = pl.loggers.CometLogger(
            save_dir=os.path.join(exp_dir, "comet_logs/"),
            experiment_key=conf["main_args"].get("comet_exp_key", None),
            log_code=True,
            log_graph=True,
            parse_args=True,
            log_env_details=True,
            log_git_metadata=True,
            log_git_patch=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_host=True,
        )
        comet_logger.log_hyperparams(conf)
        loggers.append(comet_logger)


    # Don't ask GPU if they are not available.
    gpus=  -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None   # Don't use ddp for multi-task training

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        logger=loggers,
        callbacks=callbacks,
        # checkpoint_callback=checkpoint,
        # early_stop_callback=callbacks[1],
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=None,
        limit_train_batches=1.0,  # Useful for fast experiment
        # fast_dev_run=True, # Useful for debugging
        # overfit_batches=0.001, # Useful for debugging
        gradient_clip_val=5.0,
        accumulate_grad_batches=conf["main_args"]["accumulate_grad_batches"],
        resume_from_checkpoint=resume_ckpt,
        deterministic=True,
        replace_sampler_ddp=False if conf["main_args"]["strategy"] == "multi_task" else True,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set_infos)
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    model_type = known_args.model
    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open(f"local/{model_type}.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
