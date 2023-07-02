import pandas as pd

def avg_mtr(mtr_list):
    avg_stoi = 0
    for s in mtr_list:
        avg_stoi += s
    return(avg_stoi/len(mtr_list))

def avg_mtri(mtri_list,mtr_list):
    s = 0
    for i in range(len(mtr_list)):
        s += mtr_list[i]-mtri_list[i]
    return(s/len(mtr_list))

df = pd.read_csv(r"D:\Project\SSL-pretraining-separation-main\exp\conv-noisy\results\sep_single\all_metrics.csv")
tmp = df[df['stoi']>df['input_stoi']].reset_index(drop=True)
print(tmp)

sdris = tmp['input_sdr'].tolist()
si_sdris = tmp['input_si_sdr'].tolist()

stois = tmp['stoi'].tolist()
sdrs = tmp['sdr'].tolist()
si_sdrs = tmp['si_sdr'].tolist()

print(avg_mtr(stois))
print(avg_mtr(sdrs))
print(avg_mtr(si_sdrs))
print(avg_mtri(sdris,sdrs))
print(avg_mtri(si_sdris,si_sdrs))

