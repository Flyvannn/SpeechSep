import time
import onnx
import torch
import soundfile as sf
from asteroid import ConvTasNet
import numpy as np
import onnxruntime
from convert_onnx import save_wave

model_path = r"D:\Project\SSL-pretraining-separation-main\model\ConvTasNet_Libri2Mix_sepnoisy_16k\pytorch_model.bin"
wave_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\test\mix_clean\61-70968-0003_2830-3980-0008.wav"
onnx_path = r"D:\Project\SSL-pretraining-separation-main\result\test_onnx\test_f16.onnx"


torch_model = ConvTasNet.from_pretrained(model_path)
torch_model.eval()

# read wav
mix_np, sr  = sf.read(wave_path, dtype="float32")
mix_np16, sr16  = sf.read(wave_path, dtype="float16")
print(mix_np.shape,mix_np16.shape)
print(sr,sr16)

# x = torch.randn(1, 81440,  requires_grad=True)
l = mix_np.shape[0]
# print(mix_np,sr)

wav_len = len(mix_np)/sr
print("wav len:{} s".format(wav_len))

pads = np.zeros((sr*8-l),dtype="float32")
wav_pad = np.concatenate((mix_np,pads))

x = torch.from_numpy(wav_pad)
x = x.unsqueeze(0)

torch_start = time.time()
torch_out = torch_model(x)
torch_end = time.time()
print("torch model cost time:{} s".format(torch_end-torch_start))
print("{} s wav per second".format(wav_len/(torch_end-torch_start)))


srcs_np = torch_out.squeeze(0).cpu().data.numpy()
srcs_np = srcs_np[:,:l]
save_wave(srcs_np,"result/torch/",mix_np)


onnx_model = onnx.load(onnx_path)
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
print("onnx model cost time:{}s".format(onnx_end-onnx_start))
print("{} s wav per second".format(wav_len/(onnx_end-onnx_start)))


ort_outs = np.squeeze(ort_outs)
print(ort_outs.shape,ort_outs)

ort_outs = ort_outs[:,:l]
save_wave(ort_outs,"result/onnx/",mix_np)


exit(0)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")