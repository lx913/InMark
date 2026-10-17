import torch
from diffusers import (
    AutoencoderKL,
    ConsistencyDecoderVAE,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)

import argparse
import json
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from math import comb


# you should run this notebook in the root directory of the hidden project for the following imports to work
# %cd ..
from utils.models import SecretDecoder

from mscoco_util import CoCo_Dataset

parser = argparse.ArgumentParser()
parser.add_argument("-learning_rate",type=float, default=1e-4)
parser.add_argument("-snr_gamma",type=str, default="1.0")
parser.add_argument("-epoch",type=int, default=50)
parser.add_argument("-guidance", type=float, default=7.5)
parser.add_argument("-steps", type=int, default=30)
parser.add_argument("-scheduler", type=str, default="dpms_m")
parser.add_argument("-vae", type=str, default="original")
args = parser.parse_args()

def sci_str(x, precision=0):
    s = f"{x:.{precision}e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

def set_scheduler(pipe, sampler):
    if sampler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpms_s":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpms_m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Sampler {sampler} not found.")
    return pipe

def load_vae(vae_name, torch_dtype):
    if vae_name == "original":
        return None
    if vae_name == "sd_vae_ft_mse":
        return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch_dtype)
    if vae_name == "clearvae":
        return AutoencoderKL.from_pretrained("bullhug/ClearVAE", torch_dtype=torch_dtype)
    if vae_name == "consistency_decoder":
        return ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch_dtype)
    raise ValueError(f"VAE {vae_name} not found.")



vae = load_vae(args.vae, torch_dtype=torch.float32)
pipe_kwargs = {
    "safety_checker": None,
    "torch_dtype": torch.float32,
}
if vae is not None:
    pipe_kwargs["vae"] = vae

pipe = StableDiffusionPipeline.from_pretrained("Manojb/stable-diffusion-2-1-base", **pipe_kwargs)
state_dict = torch.load(f"LoRA/checkpoint/gt_dm_watermark_lr{sci_str(args.learning_rate)}_snr{args.snr_gamma}_v1/all/unet_final.pt", map_location="cuda:0")
print("Model path: ", f"LoRA/checkpoint/gt_dm_watermark_lr{sci_str(args.learning_rate)}_snr{args.snr_gamma}_v1/all/unet_final.pt")
print("VAE: ", args.vae)
pipe.unet.load_state_dict(state_dict)
pipe = set_scheduler(pipe, args.scheduler)
pipe = pipe.to("cuda:0")
print(pipe.scheduler)

val_transforms = transforms.Compose(
        [   #! args.resolution = 512
            transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )

val_dataset = CoCo_Dataset(root = '../dataset/MSCOCO2017/val2017/',
                        annFile = '../dataset/MSCOCO2017/annotations/captions_val2017.json', transform=val_transforms)

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=64,
        num_workers=0,
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    


msg_bits = "110110011110000111011100001011100100011001110001"
msg_ori = torch.tensor([c == '1' for c in msg_bits], dtype=torch.bool, device=device).unsqueeze(0)  # (1,48)

default_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

pretrain_dict = torch.load("ckpts/pretained_latentwm.pth", map_location=device)
msgdecoder = SecretDecoder(output_size=len(msg_bits))
msgdecoder.load_state_dict(pretrain_dict['sec_decoder'])
msgdecoder = msgdecoder.to(device).eval()

def calculate_fpr(tau, k):
    sum_combinations = sum(comb(k, i) for i in range(tau + 1, k + 1))
    fpr = 1 / (2**k) * sum_combinations
    return fpr

def get_threshold(k, fpr):
    tau = 0
    while calculate_fpr(tau, k) > fpr:
        tau += 1
    return tau

tpr_shreshold = 1e-6
tau = get_threshold(len(msg_bits), tpr_shreshold)  # Adjust FPR as needed

k = msg_ori.shape[1]

def decode_one(pil_img):
    wm_image = default_transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        logits = msgdecoder(wm_image.float())                       # (1,k,2)
    decoded_bits = torch.argmax(logits, dim=-1).bool()             # (1,k)
    match = int((decoded_bits == msg_ori).sum().item())
    acc = match / k
    ok = (match >= tau)
    return acc, match, ok

all_acc, all_match, all_ok = [], [], []
TP, FN = 0, 0
save_dir = Path("syn_images") / f"lr{args.learning_rate}_snr{args.snr_gamma}_epoch{args.epoch}_guidance{args.guidance}_steps{args.steps}_scheduler{args.scheduler}_vae{args.vae}"
save_dir.mkdir(parents=True, exist_ok=True)
img_idx = 0

for i, batch in enumerate(val_dataloader):
    _, captions = batch
    images = pipe(
        prompt=list(captions),
        num_inference_steps=args.steps,      # 采样步数
        guidance_scale=args.guidance # CFG scale
    ).images

    print(len(images))
    
    for pil_img in images:
        pil_img.save(save_dir / f"{img_idx:06d}.png")
        img_idx += 1
        acc, match, ok = decode_one(pil_img)
        all_acc.append(acc)
        all_match.append(match)
        all_ok.append(ok)
        if ok:
            TP += 1
        else:
            FN += 1

    print(f"Total images: {len(all_acc)}")
    print(f"Mean bit accuracy: {sum(all_acc)/len(all_acc):.4f}")
    print(f"Pass rate (M>=tau={tau}): {sum(all_ok)/len(all_ok):.4f}")
    print(f"Mean matches M: {sum(all_match)/len(all_match):.2f} / {k}")
    print(f"TP: {TP}, FN: {FN}")
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"TPR: {tpr:.4f}")

print(f"Total images: {len(all_acc)}")
print(f"Mean bit accuracy: {sum(all_acc)/len(all_acc):.4f}")
print(f"Pass rate (M>=tau={tau}): {sum(all_ok)/len(all_ok):.4f}")
print(f"Mean matches M: {sum(all_match)/len(all_match):.2f} / {k}")
print(f"TP: {TP}, FN: {FN}")
tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
print(f"TPR: {tpr:.4f}")

results_dir = Path("saved_results") / "coco"
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / f"results_lr{args.learning_rate}_snr{args.snr_gamma}_epoch{args.epoch}_guidance{args.guidance}_steps{args.steps}_scheduler{args.scheduler}_vae{args.vae}.json"

results = {
    "learning_rate": args.learning_rate,
    "snr_gamma": args.snr_gamma,
    "epoch": args.epoch,
    "guidance": args.guidance,
    "steps": args.steps,
    "scheduler": args.scheduler,
    "vae": args.vae,
    "tau": tau,
    "total_images": len(all_acc),
    "mean_bit_accuracy": sum(all_acc) / len(all_acc),
    "pass_rate": sum(all_ok) / len(all_ok),
    "mean_matches": sum(all_match) / len(all_match),
    "k": k,
    "TP": TP,
    "FN": FN,
    "TPR": tpr,
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to: {results_path}")
