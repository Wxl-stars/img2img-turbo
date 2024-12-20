from datetime import date
from loguru import logger
import refile, json
from tqdm import tqdm

KEY = "vlm"

prefix = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/"
all_path = list(refile.smart_glob(refile.smart_path_join(prefix, f"{KEY}_day_*2024-12-18*.json")))
print(len(all_path))
all_json_data = {}

# merge
for path in tqdm(all_path):
    json_data = json.load(refile.smart_open(path))
    for key in json_data.keys():
        if key not in all_json_data:
            all_json_data[key] = []
        all_json_data[key] += json_data[key]

print(all_json_data.keys())
# save
TODAY = TODAY = str(date.today())
day_num = len(all_json_data["day"])
night_num = len(all_json_data["night"])
unkown_num = len(all_json_data["unkown"])

save_path = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/{KEY}_day_{day_num}_night_{night_num}_{TODAY}_all.json"
with refile.smart_open(save_path, "w") as f:
    json.dump(all_json_data, f, indent=2)

logger.info(f"save as {save_path}")
logger.info(f"day:{day_num}    night:{night_num}  unkown:{unkown_num}")
import IPython; IPython.embed()