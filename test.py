import soundfile as sf
import sox
def upsample_wav(file,out_path,rate):
    tfm = sox.Transformer()
    tfm.rate(rate)
    tfm.build(file, out_path)
    # sf.write(out_path)
file = r"C:\Users\14389\Desktop\mix.wav"
out_path = r"C:\Users\14389\Desktop\test_mix.wav"
rate = 16000
upsample_wav(file,out_path,rate)

exit(0)

import wave
import contextlib
fname = r'D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\dev\s1\84-121123-0004_2428-83699-0000.wav'
fname2 = r'D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\dev\s2\84-121123-0004_3000-15664-0014.wav'
fname3 = r'D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\dev\noise\84-121123-0004_3000-15664-0014.wav'


def dur(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(duration, "s")


import soundfile as sf
data, samplerate = sf.read(fname)
print(data[-10:],len(data))


import librosa
import matplotlib.pyplot as plt
def duration(fname):
    inaudio, sr = librosa.load(fname, sr=None, mono=False)
    in2, _ = librosa.load(fname2, sr=None, mono=False)
    in3, _ = librosa.load(fname3, sr=None, mono=False)

    total_dur = len(inaudio)
    x = [i for i in range(total_dur)]

    fig, ax = plt.subplots()
    ax.plot(x, inaudio,c='r')
    ax.plot(x,in2,c='b')
    ax.plot(x,in3,c='black')
    # ax.plot(x, x ** 3, label='cubic')

    ax.legend()

    # plt.plot(x,inaudio,c='b')
    # plt.plot(x2,in2,c='r')
    plt.ylim(-0.4, 0.4)  # 设置y轴的数值显示范围
    plt.show()

    duration = total_dur / sr
    return duration

print(duration(fname))