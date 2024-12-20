import json
import random
import re
import cv2
import refile
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from my_utils.training_utils import build_transform
from my_utils.training_utils import UnpairedDataset

IMG_EXT = ["png", "jpg", "jpeg"]

class PrivateUnpairedDataset(UnpairedDataset):
    def __init__(self, dataset_folder, image_prep, tokenizer, fixed_caption_src, fixed_caption_tgt):
        self.dataset_folder = dataset_folder
        self.tokenizer = tokenizer
        self.fixed_caption_src = fixed_caption_src
        self.fixed_caption_tgt = fixed_caption_tgt

        # load img path and split them
        if ".json" in dataset_folder:
            self._load_data_from_json()
        else:
            self._load_data()

        self.T = build_transform(image_prep)

        self.input_ids_src = self.tokenizer(
            self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        self.input_ids_tgt = self.tokenizer(
            self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

    def _load_data_from_json(self):
        json_data = json.load(refile.smart_open(self.dataset_folder))
        self.l_imgs_src = json_data["day"]
        if "good_night" in json_data.keys():
            self.l_imgs_tgt = json_data["good_night"]
        else:
            self.l_imgs_tgt = json_data["night"]


    def _load_data(self):
        img_paths_list = []
        for data_dir_item in self.dataset_folder:
            if data_dir_item.split(".")[-1] in IMG_EXT:
                img_paths_list.append(data_dir_item)
            for root, _, files in refile.smart_walk(data_dir_item):
                for file in files:
                    if file.split(".")[-1] in IMG_EXT:
                        cur_file_path = refile.smart_path_join(root, file)
                        img_paths_list.append(cur_file_path)

        self.l_imgs_src = []
        self.l_imgs_tgt = []

        for path in tqdm(img_paths_list, desc="[split dataset...]"):
            match = re.search(r'(\d{8}_\d{6})', path)
            datetime_part = match.group(1) if match else None
            time = datetime_part.split("_")[-1]
            if time <= "180000" and time >= "070000":
                self.l_imgs_src.append(path)  # 白天
            if time <= "050000" or time >= "190000":
                self.l_imgs_tgt.append(path)  # 夜晚
        import IPython; IPython.embed()
        
    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        # img_pil_src = Image.open(img_path_src).convert("RGB")
        # img_pil_tgt = Image.open(img_path_tgt).convert("RGB")

        # prepare source img
        img_pil_src = refile.smart_load_image(img_path_src)
        img_pil_src = cv2.cvtColor(img_pil_src, cv2.COLOR_BGR2RGB)
        img_pil_src = Image.fromarray(img_pil_src)

        # prepare target img
        img_pil_tgt = refile.smart_load_image(img_path_tgt)
        img_pil_tgt = cv2.cvtColor(img_pil_tgt, cv2.COLOR_BGR2RGB)
        img_pil_tgt = Image.fromarray(img_pil_tgt)

        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
