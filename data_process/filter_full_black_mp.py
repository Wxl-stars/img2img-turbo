from loguru import logger
import refile, json
import cv2
import os
from tqdm import tqdm
import random
from multiprocessing import Pool, Manager

# 读取 JSON 文件
json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_all.json"
# json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/vlm_day_71840_night_12025_2024-12-18_all.json"
json_data = json.load(refile.smart_open(json_path))

night_paths = json_data["night"]
output_dir_1 = "./pick_res_25"
if not os.path.exists(output_dir_1):
    os.makedirs(output_dir_1)
output_dir_2 = "./pick_res_30"
if not os.path.exists(output_dir_2):
    os.makedirs(output_dir_2)

# 创建共享的列表，用于存储符合条件的图像路径
manager = Manager()
good_night_list = manager.list()

# 定义处理单张图片的函数
def process_image(path):
    try:
        img = refile.smart_load_image(path)
        if img.mean() < 30:  # 筛选条件
            return None
        
        # # 随机保存部分图像
        # if img.mean() < 25 and random.choice([True, False, False, False, False]):
        #     name = refile.smart_path_join(output_dir_1,  str(int(img.mean())) + "_" + path.split("/")[-1])
        #     cv2.imwrite(name, img)

        #         # 随机保存部分图像
        # if img.mean() < 30 and img.mean() > 25 and random.choice([True, False, False, False, False]):
        #     name = refile.smart_path_join(output_dir_2, str(int(img.mean())) +  "_" + path.split("/")[-1])
        #     cv2.imwrite(name, img)
        
        return path
    except Exception as e:
        logger.error(f"Error processing image {path}: {e}")
        return None

# 多进程处理
with Pool(processes=120) as pool:  # 设置进程数，通常与 CPU 核数相当
    for result in tqdm(pool.imap_unordered(process_image, night_paths), desc="[Processing images...]"):
        if result is not None:
            good_night_list.append(result)

# 将结果保存到 JSON 文件
json_data["good_night"] = list(good_night_list)
good_night_num = len(json_data["good_night"])

new_json_path = json_path.replace(".json", f"good_night_{good_night_num}.json")
# with refile.smart_open(new_json_path, "w") as f:
#     json.dump(json_data, f, indent=2)

all_night_num = len(night_paths)
logger.info(f"new json save as {new_json_path}")
logger.info(f"Processed {all_night_num} images, {good_night_num} passed the filter.")
logger.info(f"keeped ratio: {good_night_num / all_night_num}")

logger.info(f"filtered num: {all_night_num - good_night_num}")
logger.info(f"filtered ratio: {(all_night_num - good_night_num)/all_night_num}")
import IPython; IPython.embed()

# e171_day_7322278_night_1706606_2024-12-18_allgood_night_1657887.json