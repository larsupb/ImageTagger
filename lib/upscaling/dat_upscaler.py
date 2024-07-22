import sys
from enum import Enum

import psutil
import torch
import torchvision.transforms as transforms
import safetensors.torch
import logging

from PIL import Image
from .DAT import DAT

TILE_SIZE = 1024

current_loaded_models = []


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception


directml_enabled = False
cpu_state = CPUState.GPU
xpu_available = False


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


def free_memory(memory_required, device, keep_loaded=None):
    if keep_loaded is None:
        keep_loaded = []
    unloaded_model = []
    can_unload = []

    for i in range(len(current_loaded_models) -1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append((sys.getrefcount(shift_model.model), shift_model.model_memory(), i))

    for x in sorted(can_unload):
        i = x[-1]
        if get_free_memory(device) > memory_required:
            break
        current_loaded_models[i].model_unload()
        unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        current_loaded_models.pop(i)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
        if mem_free_torch > mem_free_total * 0.25:
            soft_empty_cache()


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())

def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia(): #This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def load_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]

    state_dict_keys = list(state_dict.keys())
    if "layers.0.blocks.2.attn.attn_mask_0" in state_dict_keys:
        return DAT(state_dict)
    return None

@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar=None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device=output_device)
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                x = max(0, min(s.shape[-1] - overlap, x))
                y = max(0, min(s.shape[-2] - overlap, y))
                s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                ps = function(s_in).to(output_device)
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                        mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)
        output[b:b+1] = out/out_div
    return output

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                logging.warning("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False

        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)

        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def upscale_image_dat(img: Image, model_path, progress=None, device=torch.device(torch.cuda.current_device())):

    transform = transforms.ToTensor()
    in_img = transform(img).unsqueeze(0).to(device)
    sd = load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    upscale_model = load_state_dict(sd).eval()
    memory_required = module_size(upscale_model)
    memory_required += (TILE_SIZE * TILE_SIZE * 3) * in_img.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += in_img.nelement() * in_img.element_size()
    free_memory(memory_required, device)
    upscale_model.to(device)
    tile = TILE_SIZE
    overlap = 32
    oom = True
    while oom:
        try:
            s = tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap,
                                upscale_amount=upscale_model.scale, pbar=progress)
            oom = False
        except OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e
    upscale_model.cpu()
    s = torch.clamp(s, min=0, max=1.0)
    transform = transforms.ToPILImage()
    img_out = transform(s.squeeze())
    img_out_alpha = Image.new("L", img_out.size, 255)  # "L" mode for single-channel (grayscale)
    img_out = Image.merge("RGBA", (*img_out.split(), img_out_alpha))
    free_memory(8096, device, False)

    # assure that image can be saved as jpg
    img_out = img_out.convert("RGB")
    return img_out
