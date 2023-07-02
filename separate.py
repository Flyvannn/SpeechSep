import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import numpy as np
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import ConvTasNet
from asteroid.models import SuDORMRFNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import WERTracker, MockWERTracker


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, default=r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\test\mix_clean\3729-6852-0037_1995-1826-0019.wav", help="the wav file needed to be separate"
)
parser.add_argument(
    "--task",
    type=str,
    default="sep_clean",
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=16000,
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./result/",
    help="Directory in exp_dir where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):

    model_path = r"model/ConvTasNet_Libri2Mix_sepclean_16k/pytorch_model.bin"
    model = ConvTasNet.from_pretrained(model_path)
    # model_path = r"D:\Project\SSL-pretraining-separation-main\model\sudormrf\best_model.pth"
    # model = SuDORMRFNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir =  conf["out_dir"]

    print(eval_save_dir)


    torch.no_grad().__enter__()
    mix, _ = sf.read(conf['test_dir'], dtype="float32")
    print(type(mix))
    mix = torch.from_numpy(mix)
    mix_np = mix.cpu().data.numpy()

    print("mix shape", mix.unsqueeze(0).shape)
    est_sources = model(mix.unsqueeze(0))
    print("est_sources shape:", est_sources.shape)

    est_srcs = est_sources.squeeze(0).cpu().data.numpy()
    print(est_srcs.shape,est_srcs)

    for src_idx, est_src in enumerate(est_srcs):
        est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
        sf.write(
            eval_save_dir + "s{}.wav".format(src_idx + 1),
            est_src,
            conf["sample_rate"],
        )

    """
    sources_list = []
    if conf['task'] == 'sep_clean':
        s1_path = conf['test_dir'].replace("mix_clean","s1")
        s2_path = conf['test_dir'].replace("mix_clean","s2")
        
    if conf['task'] == 'enh_single':
        s1_path = conf['test_dir'].replace("mix_single","s1")
        s2_path = conf['test_dir'].replace("mix_single","noise")
        
    if conf['task'] == 'sep_noisy':
        s1_path = conf['test_dir'].replace("mix_both","s1")
        s2_path = conf['test_dir'].replace("mix_both","s2")
        
    s1, _ = sf.read(s1_path, dtype="float32")
    s2, _ = sf.read(s2_path, dtype="float32")
    sources_list.append(s1)
    sources_list.append(s2)
    sources = np.vstack(sources_list)
    sources = torch.from_numpy(sources)

    # mix, sources = tensors_to_device([mix, sources], device=model_device)
    # 
    # print("mix shape",mix.unsqueeze(0).shape)
    # est_sources = model(mix.unsqueeze(0))
    # print("est_sources shape:",est_sources.shape)


    loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
    sources_np = sources.cpu().data.numpy()
    est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
    print("shape",est_sources_np.shape)

    utt_metrics = get_metrics(
        mix_np,
        sources_np,
        est_sources_np,
        sample_rate=conf["sample_rate"],
        metrics_list=compute_metrics,
    )

    sf.write(eval_save_dir + "mixture.wav", mix_np, conf["sample_rate"])

    for src_idx, est_src in enumerate(est_sources_np):
        est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
        sf.write(
            eval_save_dir + "s{}_estimate.wav".format(src_idx+1),
            est_src,
            conf["sample_rate"],
        )
    # Write local metrics to the example folder.
    print(utt_metrics)
    with open(eval_save_dir + "metrics.json", "w") as f:
        json.dump(utt_metrics, f, indent=0)
    """

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    print(arg_dic)
    main(arg_dic)
