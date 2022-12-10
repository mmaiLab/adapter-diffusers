import math
import os
import sys
import random
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader


from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel

from adadiffusers import (
    AdapterConfig,
    AdaCLIPTextModel,
    AdaUNet2DConditionModel,
    AdaAutoencoderKL,
)

ROOT = Path('/gs/hs1/tga-i/otake.s.ad/diffusion')

TEMPLATES = [
    'training_templates_smallest',
    'reg_templates_smallest',
    'imagenet_templates_small',
    'imagenet_dual_templates_small',
    'imagenet_style_templates_small',
    'imagenet_dual_style_templates_small',
    'imagenet_class_style_templates_small',
    'style_templates',
]

DEEPFACE_EMOTIONS = [
    'sad',
    'angry',
    'surprise',
    'fear',
    'happy',
    'disgust',
    # 'neutral' unconditional
]


def get_template(id: int):
    '''
    get personalized templates
    --------------------------------
    keys: 
        0: 'training_templates_smallest'
        1: 'reg_templates_smallest'
        2: 'imagenet_templates_small'
        3: 'imagenet_dual_templates_small'
        4: 'imagenet_style_templates_small'
        5: 'imagenet_dual_style_templates_small'
        6: 'imagenet_class_style_templates_small'
        7: 'style_templates'
    '''

    training_templates_smallest = [
        'photo of a sks {}',
    ]

    reg_templates_smallest = [
        'photo of a {}',
    ]

    imagenet_templates_small = [
        'a photo of a {}',
        'a rendering of a {}',
        'a cropped photo of the {}',
        'the photo of a {}',
        'a photo of a clean {}',
        'a photo of a dirty {}',
        'a dark photo of the {}',
        'a photo of my {}',
        'a photo of the cool {}',
        'a close-up photo of a {}',
        'a bright photo of the {}',
        'a cropped photo of a {}',
        'a photo of the {}',
        'a good photo of the {}',
        'a photo of one {}',
        'a close-up photo of the {}',
        'a rendition of the {}',
        'a photo of the clean {}',
        'a rendition of a {}',
        'a photo of a nice {}',
        'a good photo of a {}',
        'a photo of the nice {}',
        'a photo of the small {}',
        'a photo of the weird {}',
        'a photo of the large {}',
        'a photo of a cool {}',
        'a photo of a small {}',
        'an illustration of a {}',
        'a rendering of a {}',
        'a cropped photo of the {}',
        'the photo of a {}',
        'an illustration of a clean {}',
        'an illustration of a dirty {}',
        'a dark photo of the {}',
        'an illustration of my {}',
        'an illustration of the cool {}',
        'a close-up photo of a {}',
        'a bright photo of the {}',
        'a cropped photo of a {}',
        'an illustration of the {}',
        'a good photo of the {}',
        'an illustration of one {}',
        'a close-up photo of the {}',
        'a rendition of the {}',
        'an illustration of the clean {}',
        'a rendition of a {}',
        'an illustration of a nice {}',
        'a good photo of a {}',
        'an illustration of the nice {}',
        'an illustration of the small {}',
        'an illustration of the weird {}',
        'an illustration of the large {}',
        'an illustration of a cool {}',
        'an illustration of a small {}',
        'a depiction of a {}',
        'a rendering of a {}',
        'a cropped photo of the {}',
        'the photo of a {}',
        'a depiction of a clean {}',
        'a depiction of a dirty {}',
        'a dark photo of the {}',
        'a depiction of my {}',
        'a depiction of the cool {}',
        'a close-up photo of a {}',
        'a bright photo of the {}',
        'a cropped photo of a {}',
        'a depiction of the {}',
        'a good photo of the {}',
        'a depiction of one {}',
        'a close-up photo of the {}',
        'a rendition of the {}',
        'a depiction of the clean {}',
        'a rendition of a {}',
        'a depiction of a nice {}',
        'a good photo of a {}',
        'a depiction of the nice {}',
        'a depiction of the small {}',
        'a depiction of the weird {}',
        'a depiction of the large {}',
        'a depiction of a cool {}',
        'a depiction of a small {}',
    ]

    imagenet_dual_templates_small = [
        'a photo of a {} with {}',
        'a rendering of a {} with {}',
        'a cropped photo of the {} with {}',
        'the photo of a {} with {}',
        'a photo of a clean {} with {}',
        'a photo of a dirty {} with {}',
        'a dark photo of the {} with {}',
        'a photo of my {} with {}',
        'a photo of the cool {} with {}',
        'a close-up photo of a {} with {}',
        'a bright photo of the {} with {}',
        'a cropped photo of a {} with {}',
        'a photo of the {} with {}',
        'a good photo of the {} with {}',
        'a photo of one {} with {}',
        'a close-up photo of the {} with {}',
        'a rendition of the {} with {}',
        'a photo of the clean {} with {}',
        'a rendition of a {} with {}',
        'a photo of a nice {} with {}',
        'a good photo of a {} with {}',
        'a photo of the nice {} with {}',
        'a photo of the small {} with {}',
        'a photo of the weird {} with {}',
        'a photo of the large {} with {}',
        'a photo of a cool {} with {}',
        'a photo of a small {} with {}',
    ]

    imagenet_style_templates_small = [
        'a painting in the style of {}',
        'a rendering in the style of {}',
        'a cropped painting in the style of {}',
        'the painting in the style of {}',
        'a clean painting in the style of {}',
        'a dirty painting in the style of {}',
        'a dark painting in the style of {}',
        'a picture in the style of {}',
        'a cool painting in the style of {}',
        'a close-up painting in the style of {}',
        'a bright painting in the style of {}',
        'a cropped painting in the style of {}',
        'a good painting in the style of {}',
        'a close-up painting in the style of {}',
        'a rendition in the style of {}',
        'a nice painting in the style of {}',
        'a small painting in the style of {}',
        'a weird painting in the style of {}',
        'a large painting in the style of {}',
    ]

    imagenet_dual_style_templates_small = [
        'a painting in the style of {} with {}',
        'a rendering in the style of {} with {}',
        'a cropped painting in the style of {} with {}',
        'the painting in the style of {} with {}',
        'a clean painting in the style of {} with {}',
        'a dirty painting in the style of {} with {}',
        'a dark painting in the style of {} with {}',
        'a cool painting in the style of {} with {}',
        'a close-up painting in the style of {} with {}',
        'a bright painting in the style of {} with {}',
        'a cropped painting in the style of {} with {}',
        'a good painting in the style of {} with {}',
        'a painting of one {} in the style of {}',
        'a nice painting in the style of {} with {}',
        'a small painting in the style of {} with {}',
        'a weird painting in the style of {} with {}',
        'a large painting in the style of {} with {}',
    ]

    imagenet_class_style_templates_small = [
        'a painting of a {} in the style of {}',
        'a rendering of a {} in the style of {}',
        'a cropped painting of a {} in the style of {}',
        'the painting of a {} in the style of {}',
        'a clean painting of a {} in the style of {}',
        'a dirty painting of a {} in the style of {}',
        'a dark painting of a {} in the style of {}',
        'a picture of a {} in the style of {}',
        'a cool painting of a {} in the style of {}',
        'a close-up painting of a {} in the style of {}',
        'a bright painting of a {} in the style of {}',
        'a cropped painting of a {} in the style of {}',
        'a good painting of a {} in the style of {}',
        'a close-up painting of a {} in the style of {}',
        'a rendition of a {} in the style of {}',
        'a nice painting of a {} in the style of {}',
        'a small painting of a {} in the style of {}',
        'a weird painting of a {} in the style of {}',
        'a large painting of a {} in the style of {}',
    ],

    style_templates = [
        'a painting in the style of {}',
        'the painting in the style of {}',
        'a photo in the style of {}',
        'the photo in the style of {}',
    ]

    emotion_templates = [
        'photo of a {} woman',
        'a woman with a {} face',
    ]

    template_dict = {
        'training_templates_smallest': training_templates_smallest,
        'reg_templates_smallest': reg_templates_smallest,
        'imagenet_templates_small': imagenet_templates_small,
        'imagenet_dual_templates_small': imagenet_dual_templates_small,
        'imagenet_style_templates_small': imagenet_style_templates_small,
        'imagenet_dual_style_templates_small': imagenet_dual_style_templates_small,
        'imagenet_class_style_templates_small': imagenet_class_style_templates_small,
        'style_templates': style_templates,
    }

    return template_dict[TEMPLATES[id]]


noise_scheduler = [
    'ddpm',
    'ddim',
    'pndm',
    'lmsd',
]

inference_prompts = [
    'photo of a {1} sks {0}',
    'a sks {0} with a {1} face',
    'photo of a sks {0}, detailed {1} face'
]

target_keys = ['learning_rate',
               'max_train_steps',
               'instance_prompt',
               'inference_prompt',
               'train_vae',
               'noise_scheduler',
               'adapter_emb_size_ratio',
               'adapter_emb_sizes',
               'context_adapter_emb_size',
               'modules',
               'blocks',
               'add_modules',
               'add_modules_blocks',
               'adapter_act',
               'context_adapter_act',
               'preln',
               'useln',
               'skip_adapter',
               'use_residual_adapter',
               'vae_blocks',
               'inception_score',
               'verification_score',
               'facial_expression_score',
               ]


@dataclass
class TrainArgs:
    pretrained_model_name_or_path: str = ''
    output_dir: Union[str, Path] = 'model'
    image_output_dir: Union[str, Path] = 'sample'
    resolution: int = 512
    center_crop: bool = True
    instance_data_root: str = 'path_to_instance_data'
    instance_prompt: str = "photo of a sks {}"
    instance_data_ratio: int = 1
    learning_rate: Union[float, dict] = 5e-05
    max_train_steps: int = 400
    train_batch_size: int = 1
    noise_scheduler: dict = field(default_factory=lambda: {
                                  'train': 'ddpm', 'infer': 'pndm'})
    gradient_accumulation_steps: int = 2
    max_grad_norm: bool = False
    mixed_precision: str = "no"  # set to "fp16" for mixed-precision training.
    # set this to True to lower the memory usage.
    gradient_checkpointing: bool = True
    seed: int = 3434554
    class_name: str = "man"
    style_name: Union[str, None] = None
    style_data_root: Union[str, None] = ''
    style_templates: Union[list, None] = field(default_factory=list)
    style_data_ratio: int = 1
    train_text_encoder: bool = False
    train_vae: Union[str, None] = 'on_rec'  # on_rec or on_ldm or None
    on_finetuning: bool = False
    adapter_path: str = ''
    run_name: str = ''


@dataclass
class ExperimentConfig:
    train_args: TrainArgs
    adapter_config: AdapterConfig


# for generating priror presavation samples
class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class ActionPromptDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            image_path = self.class_images_path[index % self.num_class_images]
            # class_images/girl/smiling_2.jpg
            action = str(image_path).split('/')[-1].split('_')[0]
            class_image = Image.open(image_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt.format(action),
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

# train_dataset = DreamBoothDataset(
#     instance_data_root=args.instance_data_dir,
#     instance_prompt=args.instance_prompt,
#     class_data_root=args.class_data_dir if args.with_prior_preservation else None,
#     class_prompt=args.class_prompt,
#     tokenizer=tokenizer,
#     size=args.resolution,
#     center_crop=args.center_crop,
# )


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # concat class and instance examples for prior preservation
    if args.with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class EmotionDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        emonet,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.emonet = emonet

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        emotion = self.emonet(
            example['instance_images'].unsqueeze(0), infer=True).emotion
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt.format(emotion[0]),
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example


class TemplateDataset(Dataset):
    '''
        Examples of argumetns
        ---------------------------------------
        class_name: 'man'
        instance_data_root: 'image/instance/'
        instance_prompt: 'photo of sks {}'
        style_data_root: 'image/style/'
        style_templates: ['a photo in the style of {}']
    '''

    def __init__(
        self,
        class_name,
        instance_data_root,
        instance_prompt,
        instance_data_ratio,
        tokenizer,
        style_name,
        style_data_root,
        style_templates,
        style_data_ratio,
        size=512,
        center_crop=False,
        instance_only=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_only = instance_only

        self.class_name = class_name
        self.style_name = style_name

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_prompt = instance_prompt

        if style_data_root is not None or instance_only:
            self.style_data_root = Path(style_data_root)
            self.style_data_root.mkdir(parents=True, exist_ok=True)
            class_images_path = list(Path(style_data_root).iterdir())

        self.style_templates = style_templates
        if instance_only:
            self.images_path = instance_images_path
        else:
            self.images_path = instance_images_path * \
                instance_data_ratio + class_images_path * style_data_ratio
        self._length = len(self.images_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_path = self.images_path[index % self._length]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if 'instance' in str(image_path) or self.instance_only:
            prompt = self.instance_prompt.format(self.class_name)
        else:
            num_descriptor = self.style_templates[0].count('{}')
            if num_descriptor == 1:
                prompt = random.choice(
                    self.style_templates).format(self.style_name)
            elif num_descriptor == 2:
                prompt = random.choice(self.style_templates).format(
                    self.class_name, self.style_name)

        example['image'] = self.image_transforms(image)
        example['prompt_ids'] = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example

    def collate_fn(self, examples):
        input_ids = [example["prompt_ids"] for example in examples]
        pixel_values = [example["image"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    def create_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


def freeze_params(module, option=None, retain_graph=False):
    param_to_update = []
    count = 0
    if option == 'all':
        for param in module.paramters():
            if retain_graph:
                param.retain_graph = True
            param.requires_grad = False
            print('All of the parameters have been frozen')

    else:
        for name, param in module.named_parameters():
            if retain_graph:
                param.retain_graph = True
            if 'adapter' in name:
                param_to_update.append(param)
                count += param.numel()
                continue
            param.requires_grad = False
        print(f'Num of adapter params {count/1e6}')
    return param_to_update


def show_params_grad(module):
    for name, param in module.named_parameters():
        if param.requires_grad:
            print(name)


def remove_graph(module):
    for param in module.paramters():
        param.detach()


def save_adapter(module, path):
    module_state_dict = module.state_dict()
    adapter_modules = [module for module in list(
        module_state_dict.keys()) if 'adapter' in module]
    adapter_state_dict = {
        adapter_module: module_state_dict[adapter_module] for adapter_module in adapter_modules}
    torch.save(adapter_state_dict, path)


def load_adapter(module, path):
    module.load_state_dict(torch.load(path), strict=False)
    return module


logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformersddbb").setLevel(logging.ERROR)


def load_base_model(adapter_config,
                    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
                    use_ada_unet=True,
                    use_ada_vae=True,
                    use_ada_text_encoder=False,
                    cache_dir='/gs/hs1/tga-i/otake.s.ad/.cache'):
    '''Loading base models
    args
    --------------------
    adapter_configs : {"adapter_config":AdapterConfig(), "ada_text_config":{'adapter_emb_size': {i:128 for i in range(12)}}}

    '''

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_auth_token=True,
        cache_dir=cache_dir
    )

    if use_ada_text_encoder:
        ada_te_config = {'adapter_emb_size': {i: 128 for i in range(12)}}
        text_encoder = AdaCLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_auth_token=True,
            cache_dir=cache_dir,
            **adapter_config.ada_text_config,
        )
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_auth_token=True,
            cache_dir=cache_dir
        )

    if use_ada_unet:
        unet = AdaUNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            use_auth_token=True,
            cache_dir=cache_dir,
            adapter_config=adapter_config
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            use_auth_token=True,
            cache_dir=cache_dir,
        )

    if use_ada_vae:
        vae = AdaAutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            use_auth_token=True,
            cache_dir=cache_dir,
            adapter_config=adapter_config
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            use_auth_token=True,
            cache_dir=cache_dir,
        )

    return {'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'vae': vae,
            'unet': unet,
            }


def merge_adapters(timestamps, merge_weight=None, device='cuda:0'):
    '''merge two adapters
    args
        timestamps: list of training timestamp
        merge_opt: option of merging adapters
        device: which device to locate modules
    '''
    model_path = ROOT / 'model'
    if not isinstance(timestamps, list):
        adapter_paths = list(model_path.glob(f'*{timestamps}*/*.pth'))
    else:
        adapter_paths = []
        for timestamp in timestamps:
            adapter_paths.append(next(model_path.glob(
                f'*{timestamp}*')) / 'unet_adapter.pth')

    adapter1 = torch.load(adapter_paths[0], map_location=device)
    adapter2 = torch.load(adapter_paths[1], map_location=device)

    merged_keys = set(adapter1.keys()) & set(adapter2.keys())
    union_keys = set(adapter1.keys()) | set(adapter2.keys())
    union_keys -= merged_keys
    merged_adapter_dict = {}
    for k in merged_keys:
        param1 = adapter1[k]
        param2 = adapter2[k]
        if merge_weight:
            merged_adapter_dict[k] = merge_weight * \
                param1.data + (1.-merge_weight) * param2.data
        else:
            merged_adapter_dict[k] = (param1.data + param2.data) / 2.
    for k in union_keys:
        if k in adapter2.keys():
            merged_adapter_dict[k] = adapter2[k]  # instance_adapter
        elif k in adapter1.keys():
            merged_adapter_dict[k] = adapter1[k]  # style_adapter

    del adapter1
    del adapter2
    with torch.no_grad():
        torch.cuda.empty_cache()

    return merged_adapter_dict
