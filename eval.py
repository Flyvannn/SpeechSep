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
    "--test_dir", type=str, default=r"./data/librimix/Libri2Mix/wav16k/min/metadata/test", help="Test directory including the csv files"
)
parser.add_argument(
    "--task",
    type=str,
    default="sep_noisy",
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="1",
    help="Directory in exp_dir where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/conv-noisy", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument("--ckpt_path", default=r"D:\Project\SSL-pretraining-separation-main\exp\conv-noisy\best_model.pth", help="Experiment checkpoint path")

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = conf['ckpt_path']
    if ".pth" in model_path or ".bin" in model_path:
        model = ConvTasNet.from_pretrained(model_path)
    else:
        model = ConvTasNet(**conf["train_conf"]["filterbank"], **conf["train_conf"]["masknet"])
        all_states = torch.load(model_path, map_location="cpu")
        state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
        model.load_state_dict(state_dict)

    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = LibriMix(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=2,
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    print(eval_save_dir)
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(save_idx):
        # Forward the network on the mixture.
        mix, sources, _ = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        print("size1:",mix.shape)

        print("size2:",mix.unsqueeze(0).shape)

        est_sources = model(mix.unsqueeze(0))
        est_sources = est_sources[:, :sources.shape[0]]

        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=compute_metrics,
        )
        utt_metrics["mix_path"] = test_set.mixture_path

        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np):
                est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)


    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)



if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    print(arg_dic)
    main(arg_dic)
