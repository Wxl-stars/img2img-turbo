import os
import argparse
from PIL import Image
import cv2
import refile
import torch
from torchvision import transforms
from tqdm import tqdm
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform

def is_img_path(path):
    extention = ["jpg", "jpeg", "png"]
    ext = path.split(".")[-1]
    if ext in extention:
        return True
    return False

def concat_images(img1, img2, direction='horizontal'):
    """
    拼接两张图片。
    
    :param img1: 第一张图片（PIL.Image.Image）
    :param img2: 第二张图片（PIL.Image.Image）
    :param direction: 拼接方向，'horizontal' 或 'vertical'
    :return: 拼接后的图片（PIL.Image.Image）
    """
    # 获取两张图片的尺寸
    width1, height1 = img1.size
    width2, height2 = img2.size

    if direction == 'horizontal':
        # 水平拼接
        new_width = width1 + width2
        new_height = max(height1, height2)
        new_image = Image.new('RGB', (new_width, new_height))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (width1, 0))
    elif direction == 'vertical':
        # 垂直拼接
        new_width = max(width1, width2)
        new_height = height1 + height2
        new_image = Image.new('RGB', (new_width, new_height))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (0, height1))
    else:
        raise ValueError("方向必须是 'horizontal' 或 'vertical'")

    return new_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_dir', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)

    img_path_list = []
    if is_img_path(args.input_image_dir):
        img_path_list.append(args.input_image_dir)
    else:
        img_path_list += list(refile.smart_glob(refile.smart_path_join(args.input_image_dir, "*.png")))
        img_path_list += list(refile.smart_glob(refile.smart_path_join(args.input_image_dir, "*.jpg")))

    for path in tqdm(img_path_list):
        input_image = refile.smart_load_image(path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        # input_image = Image.open(args.input_image).convert('RGB')
        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            if args.use_fp16:
                x_t = x_t.half()
            output = model(x_t, direction=args.direction, caption=args.prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        # save the output image
        bname = os.path.basename(path)
        os.makedirs(args.output_dir, exist_ok=True)
        result = concat_images(input_image, output_pil, direction='horizontal')
        save_path = os.path.join(args.output_dir, bname)
        result.save(save_path)
        print(f"save as {save_path}")
