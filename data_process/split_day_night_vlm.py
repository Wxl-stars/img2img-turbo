import json
import re
import refile
from tqdm import tqdm
from loguru import logger

IMG_EXT = ["png", "jpg", "jpeg"]

dataset_folder = ["s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/imgs"]

# img_paths_list = []
# for data_dir_item in dataset_folder:
#     if data_dir_item.split(".")[-1] in IMG_EXT:
#         img_paths_list.append(data_dir_item)
#     for root, _, files in tqdm(refile.smart_walk(data_dir_item)):
#         for file in files:
#             if file.split(".")[-1] in IMG_EXT:
#                 cur_file_path = refile.smart_path_join(root, file)
#                 img_paths_list.append(cur_file_path)

json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/ariadne/datasets/vlm/data/mtx/model/yolo-world/v1/vlm_train_label_data.json"

json_data = json.load(refile.smart_open(json_path))

l_imgs_src = []
l_imgs_tgt = []
unkwon_list = []
error_list = []
print(len(json_data))

for nori_id in tqdm(json_data, desc="[split dataset...]"):
    img_path = json_data[nori_id]['img_path']
    json_path = json_data[nori_id]['json_path']
    match = re.search(r'(\d{8}_\d{6})', json_path)
    datetime_part = match.group(1) if match else None
    if datetime_part is None:
        error_list.append(img_path)
        continue

    time = datetime_part.split("_")[-1]
    if time <= "180000" and time >= "070000":
        l_imgs_src.append(img_path)  # 白天
    elif time <= "050000" or time >= "190000":
        l_imgs_tgt.append(img_path)  # 夜晚
    else:
        unkwon_list.append(img_path)

json_data = dict()
json_data["day"] = l_imgs_src  # 74976
json_data["night"] = l_imgs_tgt  # 6397
json_data["unkonw"] = unkwon_list
json_data["error"] = error_list

day_num = len(l_imgs_src)
night_num = len(l_imgs_tgt)

save_path = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/vlm_day_{day_num}_night_{night_num}.json"
with refile.smart_open(save_path, "w") as f:
    json.dump(json_data, f, indent=2)

for key in json_data.keys():
    print(f"{key}: ", len(json_data[key]))

logger.info(f"save as {save_path}")

import IPython; IPython.embed()