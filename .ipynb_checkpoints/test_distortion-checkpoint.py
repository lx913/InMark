import argparse
import io
import json
from math import comb
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.io import ImageReadMode, read_image

from utils.models import SecretDecoder

from diffusers import StableDiffusionImg2ImgPipeline

import kornia as K

TARGET_SIZE = 512
MSG_BITS = "110110011110000111011100001011100100011001110001"
DECODER_CKPT = "ckpts/pretained_latentwm.pth"
FPR_THRESHOLD = 1e-6
IMAGE_PATH = "syn_images/lr0.0001_snr1.0_epoch50"
pipe = None
pipe2 = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distortion_type",
        type=str,
        required=True,
        choices=[
            "identity",
            "color_jitter",
            "crop",
            "blur",
            "noise",
            "jpeg_compress",
            "resize",
            "rotation",
            "sharpness",
            "SDEdit",
            "SDEdit2",
        ],
        help="Distortion type applied to all images in image_path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for distorted images. Defaults to <image_path>_distorted/<distortion_type>.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="saved_results/distortion",
        help="Directory for decoding statistics.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def calculate_fpr(tau, k):
    sum_combinations = sum(comb(k, i) for i in range(tau + 1, k + 1))
    return (1 / (2**k)) * sum_combinations


def get_threshold(k, fpr):
    tau = 0
    while calculate_fpr(tau, k) > fpr:
        tau += 1
    return tau


def load_image_paths(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_path does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_path must be a directory: {image_dir}")

    image_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        image_paths.extend(image_dir.glob(pattern))

    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return image_paths


def read_image_as_tensor(image_path, device):
    image = read_image(str(image_path), ImageReadMode.RGB).float() / 255.0
    image = image.unsqueeze(0).to(device)

    return image

def SDEdit(image, version=1):
    global pipe
    global pipe2
    if version == 1:
        if pipe is None:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', safety_checker=None).to('cuda')
        cur_pipe = pipe
    elif version == 2:
        if pipe2 is None:
            pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained('Manojb/stable-diffusion-2-1-base', safety_checker=None).to('cuda')
        cur_pipe = pipe2
    oimages = [transforms.ToPILImage()(image.squeeze(0).cpu())]
    images = cur_pipe(prompt="masterpiece", 
                  image=oimages,
                  strength=0.1 if version == 1 else 0.2,
                  num_inference_steps=10,
                  guidance_scale=7.5 ,
                  output_type='pt').images
    return images

def crop(encoded_image, size=(460, 460)):
    return transforms.RandomCrop(size=size)(encoded_image)


def distortion_unit(encoded_image, distortion_type):
    if distortion_type == "identity":
        distorted_image = encoded_image
    elif distortion_type == "color_jitter":
        distorted_image = encoded_image.clone()
        distorted_image = K.augmentation.ColorJiggle(
                        brightness=(0.9,1.1),
                        contrast=(0.9,1.1),
                        saturation=(0.9,1.1),
                        hue=(-0.1,0.1),
                        p=1)(distorted_image)
    elif distortion_type == "crop":
        distorted_image = crop(encoded_image)
    elif distortion_type == "blur":
        distorted_image = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_image)
    elif distortion_type == "noise":
        distorted_image = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_image)
    elif distortion_type == "jpeg_compress":
        pil_image = transforms.ToPILImage()(encoded_image.squeeze(0).cpu())
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        jpeg_image = Image.open(buffer).convert("RGB")
        distorted_image = transforms.ToTensor()(jpeg_image).unsqueeze(0).to(encoded_image.device)
    elif distortion_type == "resize":
        distorted_image = F.interpolate(
            encoded_image,
            scale_factor=(0.5, 0.5),
            mode="bicubic",
            align_corners=False,
        )
    elif distortion_type == "rotation":
        distorted_image = TF.rotate(encoded_image, angle=15.0)
    elif distortion_type == "sharpness":
        distorted_image = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_image)
    elif distortion_type == "SDEdit":
        distorted_image = SDEdit(encoded_image)
    elif distortion_type == "SDEdit2":
        distorted_image = SDEdit(encoded_image, version=2)
    else:
        raise ValueError(f"Unsupported distortion type: {distortion_type}")

    distorted_image = torch.clamp(distorted_image, 0.0, 1.0)
    distorted_image = F.interpolate(
        distorted_image,
        size=(TARGET_SIZE, TARGET_SIZE),
        mode="bicubic",
        align_corners=False,
    )
    return torch.clamp(distorted_image, 0.0, 1.0)


def decode_one_tensor(image_tensor, decoder, msg_ori, tau):
    wm_image = image_tensor * 2.0 - 1.0
    with torch.no_grad():
        logits = decoder(wm_image.float())
    decoded_bits = torch.argmax(logits, dim=-1).bool()
    match = int((decoded_bits == msg_ori).sum().item())
    k = msg_ori.shape[1]
    acc = match / k
    ok = match >= tau
    return acc, match, ok


def build_output_dir(args):
    if args.output_dir is not None:
        return Path(args.output_dir) / args.distortion_type
    image_dir = Path(args.image_path)
    return image_dir.parent / f"{image_dir.name}_distorted" / args.distortion_type


def main():
    args = parse_args()
    args.image_path = IMAGE_PATH

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    image_paths = load_image_paths(args.image_path)
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    msg_ori = torch.tensor(
        [c == "1" for c in MSG_BITS],
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    k = msg_ori.shape[1]
    tau = get_threshold(k, FPR_THRESHOLD)

    pretrain_dict = torch.load(DECODER_CKPT, map_location=device)
    msgdecoder = SecretDecoder(output_size=len(MSG_BITS))
    msgdecoder.load_state_dict(pretrain_dict["sec_decoder"])
    msgdecoder = msgdecoder.to(device).eval()

    all_acc = []
    all_match = []
    all_ok = []
    tp = 0
    fn = 0

    for image_path in tqdm(image_paths, desc=f"Applying {args.distortion_type}"):
        encoded_image = read_image_as_tensor(image_path, device)
        distorted_image = distortion_unit(encoded_image, args.distortion_type)

        save_path = output_dir / image_path.name
        distorted_pil = transforms.ToPILImage()(distorted_image.squeeze(0).cpu())
        distorted_pil.save(save_path)

        acc, match, ok = decode_one_tensor(distorted_image, msgdecoder, msg_ori, tau)
        all_acc.append(acc)
        all_match.append(match)
        all_ok.append(ok)
        if ok:
            tp += 1
        else:
            fn += 1

    pass_rate = sum(all_ok) / len(all_ok)
    mean_bit_accuracy = sum(all_acc) / len(all_acc)
    mean_matches = sum(all_match) / len(all_match)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results = {
        "image_path": str(Path(args.image_path).resolve()),
        "distortion_type": args.distortion_type,
        "output_dir": str(output_dir.resolve()),
        "decoder_ckpt": DECODER_CKPT,
        "target_size": TARGET_SIZE,
        "tau": tau,
        "total_images": len(all_acc),
        "mean_bit_accuracy": mean_bit_accuracy,
        "pass_rate": pass_rate,
        "mean_matches": mean_matches,
        "k": k,
        "TP": tp,
        "FN": fn,
        "TPR": tpr,
    }

    image_dir_name = Path(args.image_path).name
    results_path = results_dir / f"{image_dir_name}_{args.distortion_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Distorted images saved to: {output_dir}")
    print(f"Results saved to: {results_path}")
    print(f"Total images: {len(all_acc)}")
    print(f"Mean bit accuracy: {mean_bit_accuracy:.4f}")
    print(f"Pass rate (M>=tau={tau}): {pass_rate:.4f}")
    print(f"Mean matches M: {mean_matches:.2f} / {k}")
    print(f"TP: {tp}, FN: {fn}, TPR: {tpr:.4f}")


if __name__ == "__main__":
    main()
