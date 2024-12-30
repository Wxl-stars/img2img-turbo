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
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 提供分类提示词
prompts = [
    "This is a self-driving scene during the day.",
    "This is a self-driving scene at dusk."
    ]

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
    
    # display = np.array(image)
    # cv2.putText(display, f"goodday: {probs[0][0]}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(255, 0, 0), thickness=5)
    # cv2.putText(display, f"dusk: {probs[0][1]}", (100, 300),cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 0), thickness=5)
    # cv2.imwrite("test.png", display[:, :, ::-1])
    # 返回最高得分的类别
    if probs[0][0] > probs[0][1] and probs[0][0] > 0.85:
        label = "goodday"
    elif probs[0][1] > probs[0][0] and probs[0][1] > 0.8:
        label = "dusk"
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
# 示例
# img_dir = "/gpfs/shared_files/wheeljack/wuxiaolei/projs/img2img-turbo/src_test/"
# img_path_list = refile.smart_glob(refile.smart_path_join(img_dir, "*.png"))
TODAY = TODAY = str(date.today())
dataset_folder = [
    # "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/images",
    "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/imgs"
    ]
# img_path_list = find_all_img_files(dataset_folder)
# logger.info(f"Total {len(img_path_list)} datas.")
# save_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/vlm_all_img_data.json"
# with refile.smart_open(save_path, "w") as f:
#     json.dump(img_path_list, f, indent=2)

images_json = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_allgood_night_1013194.json"
json_data = json.load(refile.smart_open(images_json))

# front_path_list = []
# for path in tqdm(img_path_list):
#     if "cam_front_120" in path:
#         front_path_list.append(path)
# del img_path_list

# for vlm 
# multi-process

if args.rank_id is not None and args.world_size is not None:
    step = args.world_size
    begin = args.rank_id
else:
    step = 1
    begin = 0

for idx in tqdm(range(begin, len(json_data["day"])-1, step), desc="[processing...]"):
    image_path = json_data["day"][idx]
    result, probs = classify_day_night_clip(image_path)
    img_suffix = image_path.split("/")[-1]
    # print(f"The image [{img_suffix}] is classified as: {result}, prob:{probs}")
    if result not in json_data.keys():
        json_data[result] = []
    json_data[result].append(image_path)

for key in json_data.keys():
    print(f"{key}: ", len(json_data[key]))

day_num = len(json_data["day"])
night_num = len(json_data["night"])
# save_path = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/vlm_day_{day_num}_night_{night_num}_{TODAY}_{args.rank_id}.json"
save_path = image_path.replace(".json", f"_goodday_{args.rank_id}")
with refile.smart_open(save_path, "w") as f:
    json.dump(json_data, f, indent=2)

for key in json_data.keys():
    print(f"{key}: ", len(json_data[key]))

# logger.info(f"process total: {len(json_data["day"])}")
# logger.info(f"day:")

logger.info(f"save as {save_path}")


