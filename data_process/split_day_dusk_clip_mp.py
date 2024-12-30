from datetime import date
import os
import cv2
import refile, json
from multiprocessing import Pool, Manager
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from loguru import logger

TODAY = TODAY = str(date.today())
# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

def classify_image(image_path):
    """
    对单张图片进行分类（白天或夜晚）。
    """
    try:
        # 打开图片并进行预处理
        image = load_form_s3(image_path)
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

        # 模型推理
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # 返回最高得分的类别
        if probs[0][0] > probs[0][1] and probs[0][0] > 0.85:
            predicted_label = "goodday"
        elif probs[0][1] > probs[0][0] and probs[0][1] > 0.8:
            predicted_label = "night"
        else:
            predicted_label = "unkown"
        print(f"img path: {image_path}, label: {predicted_label}")
        return image_path, predicted_label
    except Exception as e:
        return image_path, f"Error: {e}"

def process_images_in_batch(image_paths, results):
    """
    多进程处理一批图片。
    """
    for image_path in image_paths:
        results.append(classify_image(image_path))

def main(image_paths, num_processes=4, images_json=None):
    """
    主函数：多进程分类图片并汇聚结果。
    """
    # # 获取图片路径列表
    # image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 创建Manager对象用于多进程间共享结果
    with Manager() as manager:
        results = manager.list()  # 用于存储分类结果
        # 将图片路径分割为多份，分配给不同的进程
        chunk_size = len(image_paths) // num_processes
        chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

        # 启动多进程
        with Pool(num_processes) as pool:
            pool.starmap(process_images_in_batch, [(chunk, results) for chunk in chunks])

        # 汇聚结果
        final_results = list(results)

    # 打印或保存最终结果
    json_data = dict()
    for image_path, label in final_results:
        print(f"Image: {image_path}, Classified as: {label}")
        if label not in json_data:
            json_data[label] = []
        json_data[label].append(image_path)

    for key in image_path_list:
        logger.info(f"{key}: {len(image_path_list[key])}")
    
    day_num = len(json_data["day"])
    night_num = len(json_data["night"])
    # save_path = f"s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_{day_num}_night_{night_num}_{TODAY}_good.json"
    goodday_num = len(json_data["goodday"])
    save_path = image_path.replace(".json", f"_goodday_{goodday_num}.json")
    with refile.smart_open(save_path, "w") as f:
        json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    # images_json = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_all_img_data.json"
    # image_path_list = json.load(refile.smart_open(images_json))
    # print("total imgs: ", len(image_path_list))  # 53421085
    # front_path_list = []
    # for path in tqdm(image_path_list):
    #     if "cam_front_120" in path:
    #         front_path_list.append(path)
    images_json = "s3://sdagent-shard-bj-baiducloud/wheeljack/wuxiaolei/img_translation/data/e171_day_7322278_night_1706606_2024-12-18_allgood_night_1013194.json"
    image_path_list = json.load(refile.smart_open(images_json))
    for key in image_path_list:
        logger.info(f"{key}: {len(image_path_list[key])}")
    day_images_list = image_path_list["day"]
    print("front day images: ", len(day_images_list))  # 9283106
    # 只处理前视图
    main(day_images_list, 20, images_json)