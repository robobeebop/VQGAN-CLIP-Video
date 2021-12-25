import gc
import io
import cv2
import sys
import PIL
import math
import lpips
import torch
import requests
import numpy as np

sys.path.append('./CLIP')
sys.path.append('./taming-transformers')

from os import path
from PIL import Image
from glob import glob
from CLIP import clip
from pathlib import Path
from IPython import display
from torch import nn, optim
from google.colab import output
from omegaconf import OmegaConf
from torchvision import transforms
from torch.nn import functional as F
from tqdm.notebook import tqdm, trange
from taming.models import cond_transformer, vqgan
from torchvision.transforms import functional as TF

def reduce_res(res, max_res_value=4.5e5, max_res_scale=1.): # max limit aprx 700x700 = 49e4
  x1, y1 = res
  if x1 * y1 < max_res_value:
    return x1, y1
  x = (max_res_value**(1/2)) / (x1/y1)**(1/2)
  return int(max_res_scale*x1*x/y1), int(max_res_scale*x)

def sinc(x):
  return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)

clamp_with_grad = ClampWithGrad.apply

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 3)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def save_img(a, dir):
    PIL.Image.fromarray(np.uint8(np.clip(a, 0, 255))).save(dir)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super().__init__()

			def forward(self, tenInput):
				tenBlue = (tenInput[:, 0:1, :, :] - 0.406) / 0.225
				tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
				tenRed = (tenInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tenRed, tenGreen, tenBlue ], 1)

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super().__init__()

				self.netBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			

			def forward(self, tenInput):
				return self.netBasic(tenInput)
			
		self.netPreprocess = Preprocess()
		self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + arguments_strModel + '.pytorch', file_name='spynet-' + arguments_strModel).items() })
	
	def forward(self, tenOne, tenTwo):
		tenFlow = []

		tenOne = [ self.netPreprocess(tenOne) ]
		tenTwo = [ self.netPreprocess(tenTwo) ]

		for intLevel in range(5):
			if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
				tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
				tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
			
		tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])

		for intLevel in range(len(tenOne)):
			tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
		

		return tenFlow

torch.backends.cudnn.enabled = True
arguments_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'
backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)

netNetwork = None
def estimate(tenOne, tenTwo):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().cuda().eval()

	assert(tenOne.shape[1] == tenTwo.shape[1])
	assert(tenOne.shape[2] == tenTwo.shape[2])

	intWidth = tenOne.shape[2]
	intHeight = tenOne.shape[1]

	tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()

def calc_opflow(img1, img2):
    img1 = PIL.Image.fromarray(img1)
    img2 = PIL.Image.fromarray(img2)

    tenFirst = torch.FloatTensor(
        np.ascontiguousarray(
            np.array(img1)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
            * (1.0 / 255.0)
        )
    )
    tenSecond = torch.FloatTensor(
        np.ascontiguousarray(
            np.array(img2)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
            * (1.0 / 255.0)
        )
    )

    tenOutput = estimate(tenFirst, tenSecond)
    return tenOutput

def get_opflow_image(np_prev_img, frame, np_img, blendflow, blendstatic):
    np_prev_img = np.float32(np_prev_img)
    frame = np.float32(frame)
    np_img = np.float32(np_img)

    h, w, _ = np_prev_img.shape

    flow = calc_opflow(np.uint8(np_prev_img), np.uint8(np_img))
    flow = np.transpose(np.float32(flow), (1, 2, 0))
    inv_flow = flow
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    framediff = (np_img*(1-blendflow) + frame*blendflow) - np_prev_img
    framediff = cv2.remap(framediff, flow, None, cv2.INTER_LINEAR)
    framediff = cv2.GaussianBlur(framediff, (5,5), 0)
    frame_flow = np_img + framediff

    magnitude, angle = cv2.cartToPolar(inv_flow[...,0], inv_flow[...,1])
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    ret, mask = cv2.threshold(norm_mag, 6, 255, cv2.THRESH_BINARY)
    flow_mask = mask.astype(np.uint8).reshape((h, w, 1))
    frame_flow_masked = cv2.bitwise_and(frame_flow, frame_flow, mask=flow_mask)

    background_blendimg = cv2.addWeighted(np_img, (1-blendstatic), frame, blendstatic, 0)
    background_masked =  cv2.bitwise_and(background_blendimg, background_blendimg, mask=cv2.bitwise_not(flow_mask))

    return frame_flow_masked, background_masked
