from thop import profile
from torchsummaryX import summary
import torch
from asteroid import ConvTasNet
import asteroid
model = ConvTasNet.from_pretrained(r"D:\Project\SSL-pretraining-separation-main\model\dprnn\best_model.pth")
x = torch.randn(1, 128000,  requires_grad=True)
flops, params = profile(model, (x, ))
print("flops:{} params:{}".format(flops,params))
print(summary(model, x))