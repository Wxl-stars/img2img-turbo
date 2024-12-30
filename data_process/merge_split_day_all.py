import argparse
from datetime import date
from loguru import logger
import refile, json
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(
    description="Scripts to generate data for vlm using AnyDoor"
)
parser.add_argument("--date", type=str, required=True, default=None)
args = parser.parse_args()

prefix = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/{args.date}"
all_path = list(refile.smart_glob(refile.smart_path_join(prefix, "*json")))
import IPython; IPython.embed()
print(len(all_path))
all_json_data = {}
pop_key = ["day", "night", "unkown", "good_night"]


# merge
for path in tqdm(all_path):
    json_data = json.load(refile.smart_open(path))    
    for key in json_data.keys():
        if key in pop_key:
            continue
        if key not in all_json_data:
            all_json_data[key] = []
        all_json_data[key] += json_data[key]

print(all_json_data.keys())
# save
TODAY = TODAY = str(date.today())
# day_num = len(all_json_data["day"])
# night_num = len(all_json_data["night"])
# unkown_num = len(all_json_data["unkown"])
# goodday_num = len(all_json_data["goodday"])
# dusk_num = len(all_json_data["dusk"])
# save_path = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/{KEY}_day_{day_num}_night_{night_num}_{TODAY}_all_split.json"
# 正则表达式
pattern = r"^([a-zA-Z_]+(?:_[a-zA-Z_]+)*)_\d+\.json$"
pattern = r"(s3://.*?/data/\d{4}-\d{2}-\d{2}/.*?)(?:_\d+\.json)"
match = re.search(pattern, all_path[0])
save_path = match.group(1) + "_all.json"
with refile.smart_open(save_path, "w") as f:
    json.dump(all_json_data, f, indent=2)
logger.info(f"json saved as {save_path}")

for key in all_json_data.keys():
    logger.info(f"{key}: {len(all_json_data[key])}")

# logger.info(f"save as {save_path}")
# logger.info(f"day:{day_num}    night:{night_num}  unkown:{unkown_num}")
import IPython; IPython.embed()