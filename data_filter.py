import os
import shutil
import json
import pandas as pd

def filter_wav(csv_path):
    df = pd.read_csv(csv_path)
    tmp = df[df['length']<16000*8].reset_index(drop=True)
    print(tmp)
    save_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\train\\"+csv_path.split("\\")[-1]
    tmp.to_csv(save_path,index=False)

# csv_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\metadata\mixture_train-100_mix_single.csv"
# filter_wav(csv_path)


def concat_csv(csv_path1,csv_path2,save_path):
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    print(df1)
    print(df2)
    df = pd.concat([df1,df2]).reset_index(drop=True)
    print(df,len(df))
    df.to_csv(save_path,index=False)

# csv1 = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\train\mixture_train-100_mix_clean.csv"
# csv2 = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\train\mixture_train-360_mix_clean.csv"
# save_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\train\mixture_train_mix_clean.csv"
# concat_csv(csv1,csv2)

def copy_file(csv_path):
    df = pd.read_csv(csv_path)
    # wav_paths = df["mixture_path"].tolist()
    save_path = df["mixture_path"].tolist()[0].split("\\")[:-1]
    save_path = "\\".join(save_path).replace("train-100","train").replace("train-360","train")
    df["new_mixture_path"] = df["mixture_path"].apply(lambda x:shutil.copy(x, save_path))
    # print(save_path)
    # for wav_path in wav_paths:
        # print(wav_path)
        # shutil.copy(wav_path, save_path)
    # df["new_mixture_path"] = df["mixture_path"].apply(lambda x:"\\".join(x.split("\\")[:-1]).replace("train-100","train").replace("train-360","train"))

# csv_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\train\mixture_train_mix_single.csv"
# copy_file(csv_path)

def revise_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["mixture_path"] = df["mixture_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k","/home/admin/zzg/dataset/LibriMix/Libri_2_8k_min_mixclean/Libri2Mix/wav16k").replace("\\","/"))
    df["source_1_path"] = df["source_1_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k","/home/admin/zzg/dataset/LibriMix/Libri_2_8k_min_mixclean/Libri2Mix/wav16k").replace("\\","/"))
    # df["source_2_path"] = df["source_2_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k","/home/admin/zzg/dataset/LibriMix/Libri_2_8k_min_mixclean/Libri2Mix/wav16k").replace("\\","/"))
    df["noise_path"] = df["noise_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k","/home/admin/zzg/dataset/LibriMix/Libri_2_8k_min_mixclean/Libri2Mix/wav16k").replace("\\","/"))
    df.to_csv(r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\train\mixture_train_mix_single.csv",index=False)
    # df["mixture_path"] = df["mixture_path"].apply(lambda x: x.replace("train-100","train").replace("train-360","train"))
    # print(df["mixture_path"].tolist())

# csv_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\metadata\mixture_train_mix_single.csv"
# revise_csv(csv_path)

def revise_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["mixture_path"] = df["mixture_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min","E:\Fly\Speech\dataset"))
    df["source_1_path"] = df["source_1_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min","E:\Fly\Speech\dataset"))
    # df["source_2_path"] = df["source_2_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min","E:\Fly\Speech\dataset"))
    df["noise_path"] = df["noise_path"].apply(lambda x: x.replace("D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min","E:\Fly\Speech\dataset"))
    df.to_csv(r"C:\Users\14389\Desktop\train\mixture_train_mix_single.csv",index=False)

csv_path = r"D:\Project\SSL-pretraining-separation-main\data\librimix\Libri2Mix\wav16k\min\metadata\train\mixture_train_mix_single.csv"
revise_csv(csv_path)

exit(0)
s = 0
csv_path = r"D:\Project\SSL-pretraining-separation-main\data\librimix\Libri2Mix\wav16k\min\metadata\test\mixture_test_mix_both.csv"
df = pd.read_csv(csv_path)
l = df["length"].tolist()
print(l)
for i in l:
    s += i/16000
print(s/len(l))
