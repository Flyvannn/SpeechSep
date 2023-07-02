import argparse

from os.path import basename, join, splitext
# 需要事先安装好 librosa 库: python3 -m pip install librosa
import librosa
parser = argparse.ArgumentParser()
parser.add_argument("in_audio", type=str, help="path of input audio file")
parser.add_argument("out_dir", type=str, help="path for storing output audio files")
parser.add_argument("duration", type=float, default=4.0, help="duration in seconds of each segment")
args = parser.parse_args()

assert args.duration > 0.0, "duration (=%fs) must be longer than 0.0s" % args.duration

file_name = splitext(basename(args.in_audio))[0]
inaudio, sr = librosa.load(args.in_audio, sr=None, mono=False)
total_dur = len(inaudio)
seg_dur = int(args.duration * sr)
if total_dur <= seg_dur:
    print(
        "The input audio is shorter than the specified segment duration. Nothing to do!"
    )
else:
    for n in range(total_dur // seg_dur):
        start = n * seg_dur
        end = start + seg_dur
        if end > total_dur:
            print("Warning: Last 1 position not reached (audio shorter than expected)")
        # librosa 只支持输出为 wav 格式
        librosa.output.write_wav(
            join(args.out_dir, file_name + "%04d.wav" % (n + 1)),
            inaudio[start:end],
            sr=sr,
            norm=False,
        )