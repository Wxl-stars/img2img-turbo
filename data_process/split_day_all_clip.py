import argparse
import random
from datetime import date
import json
import cv2
from loguru import logger
import numpy as np
import torch
import refile
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

IMG_EXT = ["png", "jpg", "jpeg"]
# 加载 CLIP 模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 提供分类提示词
prompts = [
    "This is a self-driving scene on sunny day.",
    "This is a self-driving scene on cloudy day.",
    "This is a self-driving scene at dusk.",
    "This is a self-driving scene on rainy day.",
    "This is a self-driving scene on snowy day.",
    "This is a self-driving scene on foggy day.",
    ]
whether_labels = ["goodday", "cloudy", "dusk", "rainy", "snowy", "foggy"]

def load_form_s3(img_path):
    img = refile.smart_load_image(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def classify_day_night_clip(image_path):
    # 读取图片
    # image = Image.open(image_path).convert("RGB")
    image = load_form_s3(image_path)

    # 预处理
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 计算图文匹配分数
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像对文本的相似度
    probs = logits_per_image.softmax(dim=1)  # 转为概率分布
    
    # # 可视化
    # display = np.array(image)
    # for idx in range(len(whether_labels)):
    #     text = whether_labels[idx] + f": {probs[0][idx]:.2f}"
    #     cv2.putText(display, text, (100, 200+idx*100), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 0, 0), thickness=5)
    
    # display = np.array(image)
    # cv2.putText(display, f"goodday: {probs[0][0]}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 0, 0), thickness=5)
    # cv2.putText(display, f"dusk: {probs[0][1]}", (100, 300),cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 0), thickness=5)
    # cv2.imwrite("test.png", display[:, :, ::-1])

    # 获取最匹配的标签
    max_prob_index = torch.argmax(probs[0]).item()
    if probs[0][max_prob_index] > 0.85:
        label = whether_labels[max_prob_index]
    else:
        label = "unkown"
    return label, probs

def find_all_img_files(dataset_folder):
    img_paths_list = []
    for data_dir_item in tqdm(dataset_folder):
        if data_dir_item.split(".")[-1] in IMG_EXT:
            img_paths_list.append(data_dir_item)
        for root, _, files in tqdm(refile.smart_walk(data_dir_item)):
            for file in tqdm(files, desc="[process files]"):
                cur_file_path = refile.smart_path_join(root, file)
                if file.split(".")[-1] in IMG_EXT:
                    img_paths_list.append(cur_file_path)
    return img_paths_list

parser = argparse.ArgumentParser(
    description="Scripts to generate data for vlm using AnyDoor"
)
parser.add_argument("--rank_id", type=int, required=False, default=0)
parser.add_argument("--world_size", type=int, required=False, default=1)
args = parser.parse_args()

TODAY = str(date.today())
PRE_FIX = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/{TODAY}"

images_json = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_allgood_night_1013194.json"
json_data = json.load(refile.smart_open(images_json))

# multi-process
if args.rank_id is not None and args.world_size is not None:
    step = args.world_size
    begin = args.rank_id
else:
    step = 1
    begin = 0

import random
random.shuffle(json_data["day"])

for idx in tqdm(range(begin, len(json_data["day"])-1, step), desc="[processing...]"):
    image_path = json_data["day"][idx]
    try:
        result, probs = classify_day_night_clip(image_path)
    except:
        continue
    img_suffix = image_path.split("/")[-1]
    # print(f"The image [{img_suffix}] is classified as: {result}, prob:{probs}")
    if result not in json_data.keys():
        json_data[result] = []
    json_data[result].append(image_path)

for key in json_data.keys():
    print(f"{key}: ", len(json_data[key]))

save_path = refile.smart_path_join(PRE_FIX, f"day_nigh_dusk_raniy_snowy_foggy_goodday_goodnight_{args.rank_id}.json")
with refile.smart_open(save_path, "w") as f:
    json.dump(json_data, f, indent=2)

for key in json_data.keys():
    print(f"{key}: ", len(json_data[key]))

logger.info(f"save as {save_path}")


