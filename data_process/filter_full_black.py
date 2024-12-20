from loguru import logger
import refile, json
import cv2
from tqdm import tqdm
import random

json_path = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_all.json"
json_data = json.load(refile.smart_open(json_path))

night_paths = json_data["night"]
json_data["good_night"] = []

# res = 0
# min_res = 9.18285815329218
for path in tqdm(night_paths, desc="[process img...]"):
    img = refile.smart_load_image(path)
    if img.mean() < 20:
        continue
    json_data["good_night"].append(path)
    if random.choice([True, False, False, False, False]):
        name = refile.smart_path_join("./pick_res", path.split("/")[-1])
        cv2.imwrite(name, img)

good_night_num = len(json_data["good_night"])
all_night_num = len(json_data["night"])
new_json_path = json_path.replace(".json", f"good_night_{good_night_num}.json")
with refile.smart_open(new_json_path, "w") as f:
    json.dump(json_data, f, indent=2)
logger.info(f"all night: {all_night_num}    good_night: {good_night_num}        ratio: {good_night_num / all_night_num}")
logger.info(f"save as {new_json_path}")
