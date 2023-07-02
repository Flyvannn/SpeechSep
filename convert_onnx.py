import torch
from asteroid import ConvTasNet
import soundfile as sf
import numpy as np

def save_wave(est_srcs,save_path,mix_np):
    for src_idx, est_src in enumerate(est_srcs):
        print(np.max(np.abs(est_src)))
        est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
        sf.write(
            save_path + "s{}.wav".format(src_idx + 1),
            est_src,
            16000,
        )

model_path = r"D:\Project\SSL-pretraining-separation-main\model\ConvTasNet_Libri2Mix_sepnoisy_16k\pytorch_model.bin"
wave_path = r"D:\FC-Audio\AudioDataset\LibriMix\Libri2Mix\wav16k\min\test\mix_clean\3729-6852-0037_1995-1826-0019.wav"
save_path = r"result/test_onnx/"

torch_model = ConvTasNet.from_pretrained(model_path)
torch_model.eval()

# mix_np, _  = sf.read(wave_path, dtype="float32")
x = torch.randn(1, 128000,  requires_grad=True)
# x = torch.from_numpy(mix_np)
# x = x.unsqueeze(0)
print(x.shape)

torch_out = torch_model(x)

# est_srcs = torch_out.squeeze(0).cpu().data.numpy()
# save_wave(est_srcs,save_path)

print(torch_out)

# Export the model
torch.onnx.export(torch_model,                       # model being run
                  x,                                 # model input (or a tuple for multiple inputs)
                  save_path + "test.onnx",                       # where to save the model (can be a file or file-like object)
                  export_params=True,                # store the trained parameter weights inside the model file
                  opset_version=11,                  # the ONNX version to export the model to
                  do_constant_folding=True,          # whether to execute constant folding for optimization
                  input_names = ['input'],           # the model's input names
                  output_names = ['output'],         # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},       # variable lenght axes
                                'output' : {0 : 'batch_size'}})
