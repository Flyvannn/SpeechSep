import time
import onnx
import torch
import soundfile as sf
from asteroid import ConvTasNet
import numpy as np
import onnxruntime
from convert_onnx import save_wave


def run_onnx_model(onnx_model,wav_path):
    # read wav
    mix_np, sr = sf.read(wav_path, dtype="float32")
    # x = torch.randn(1, 81440,  requires_grad=True)
    l = mix_np.shape[0]
    # print(mix_np,sr)

    wav_len = len(mix_np) / sr
    print("wav len:{} s".format(wav_len))

    pads = np.zeros((sr * 8 - l), dtype="float32")
    wav_pad = np.concatenate((mix_np, pads))

    x = torch.from_numpy(wav_pad)
    x = x.unsqueeze(0)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(onnx_model.graph))
    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

    onnx_start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_end = time.time()
    print("onnx model cost time:{}s".format(onnx_end - onnx_start))
    print("{} s wav per second".format(wav_len / (onnx_end - onnx_start)))

    ort_outs = np.squeeze(ort_outs)
    print(ort_outs.shape, ort_outs)

    ort_outs = ort_outs[:, :l]


if __name__ == '__main__':
    onnx_path = r"D:\Project\SSL-pretraining-separation-main\result\test_onnx\test.onnx"
    onnx_model = onnx.load(onnx_path)
    wav_path = r"C:\Users\14389\Desktop\test_mix.wav"
    run_onnx_model(onnx_model, wav_path)