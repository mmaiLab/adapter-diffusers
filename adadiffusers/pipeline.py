import os
import re
import inspect
import omegaconf
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from pathlib import Path
from omegaconf import OmegaConf
from copy import deepcopy

import torch
from torch import autocast
from transformers import CLIPTokenizer, CLIPTextModel
from adadiffusers import (
    AdaUNet2DConditionModel,
    AdaAutoencoderKL,
)

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler
# from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput


from .utils import load_base_model, inference_prompts, target_keys
from tqdm.notebook import tqdm
from deepface import DeepFace

from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
from PIL import Image

SCHEDULERS = {
    'ddpm': DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
    'ddim': DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
    'pndm': PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
    'lmsd': LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
}

os.environ['DEEPFACE_HOME'] = "/gs/hs1/tga-i/otake.s.ad/.cache"
os.environ['HF_DATASET_CACHE'] = "/gs/hs1/tga-i/otake.s.ad/.cache"


class GenerateImagePipeline(object):
    def __init__(self, init_config, config_instance=None, config_style=None, merge_weight=None, device='cuda:0', use_adapter=True, instance_adapter=None, style_adapter=None):

        self.config = init_config
        self.device = device

        args = self.config.train_args
        adapter_config = self.config.adapter_config

        if use_adapter:
            models = load_base_model(adapter_config,
                                     use_ada_vae=True if args.train_vae else False,
                                     use_ada_text_encoder=False)
        else:
            models = load_base_model(None,
                                     use_ada_unet=False,
                                     use_ada_vae=False,
                                     use_ada_text_encoder=False)

        tokenizer = models['tokenizer']
        text_encoder = models['text_encoder']
        unet = models['unet']
        vae = models['vae']

        self.output_dir = Path(args.output_dir)
        self.image_dir = Path(args.image_output_dir)

        if args.train_vae:
            vae.load_state_dict(torch.load(
                self.output_dir / 'vae_adapter.pth', map_location=device), strict=False)
#         if args.train_text_encoder:
#             text_encoder.load_state_dict(torch.load(self.output_dir / 'text_encoder.pth', map_location=device), strict=False)

        if config_instance:
            model_path = config_instance.train_args.output_dir / 'unet_adapter.pth'
            unet.load_state_dict(torch.load(
                model_path, map_location=device), strict=False)
        else:
            if merge_weight:
                unet.load_state_dict(self.merge_adapters(
                    merge_weight=merge_weight, instance_adapter=instance_adapter, style_adapter=style_adapter), strict=False)
            else:
                unet.load_state_dict(torch.load(
                    self.output_dir / 'unet_adapter.pth', map_location=device), strict=False)

        if config_style:
            model_path = config_style.train_args.output_dir / 'unet_adapter.pth'
            unet.load_state_dict(torch.load(
                model_path, map_location=device), strict=False)

        self.pipeline = MyPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=SCHEDULERS[args.noise_scheduler['infer']],
        ).to(device)

    def merge_adapters(self, merge_weight=None, instance_adapter=None, style_adapter=None):
        '''merge two adapters
        args
            merge_weight: weight for parameters of the instance adapter
        '''
        adapter_paths = list(self.output_dir.glob(f'*.pth'))

        if style_adapter:
            adapter1 = torch.load(
                self.output_dir / style_adapter, map_location=self.device)
        else:
            adapter1 = torch.load(adapter_paths[0], map_location=self.device)
        if instance_adapter:
            adapter2 = torch.load(
                self.output_dir / instance_adapter, map_location=self.device)
        else:
            adapter2 = torch.load(adapter_paths[1], map_location=self.device)

        merged_keys = set(adapter1.keys()) & set(adapter2.keys())
        union_keys = set(adapter1.keys()) | set(adapter2.keys())
        union_keys -= merged_keys
        merged_adapter_dict = {}
        for k in merged_keys:
            param1 = adapter1[k]
            param2 = adapter2[k]
            merged_adapter_dict[k] = (
                1.-merge_weight) * param1.data + merge_weight * param2.data
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

    def __call__(self, prompt, seed=42, num_samples=1, num_rows=2, num_inference_steps=50, guidance_scale=7.5):

        all_images = []
        generator = torch.Generator(device=self.device).manual_seed(seed)

        for _ in range(num_rows):
            with autocast("cuda"):
                images = self.pipeline([prompt] * num_samples, num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale, generator=generator)
                all_images.extend(images)

        return all_images

    def gen_emotional_image(self, base_prompt, emotion=None, num_images=100, num_samples=2, num_inference_steps=50, guidance_scale=7.5):
        if emotion is not None:
            prompt = base_prompt.format(emotion)
            output_dir = self.image_dir / emotion
        else:
            prompt = base_prompt
            output_dir = self.image_dir / 'no_sks'

        if base_prompt == '':
            prompt = ''
            output_dir = self.image_dir / 'no_cond'

        all_images = []
        total_images = 0
        count = 0
        current_images = len(list(output_dir.glob('*png')))
        total_images = current_images  # for continous processing
        count = total_images
        output_dir.mkdir(exist_ok=True, parents=True)
        while total_images < num_images:
            generator = torch.Generator(
                device=self.device).manual_seed(total_images)
            total_images += num_samples
            with autocast("cuda"):
                images = self.pipeline([prompt] * num_samples, num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale, generator=generator)
                all_images.extend(images)

            for img in all_images:
                count += 1
                filename = f'{count}.png'
                img.save(output_dir / filename)
            all_images = []


@dataclass
class EvaluateMetric:
    image_dir: Union[str, Path]
    inference_prompt: str
    learning_rate: float
    max_train_steps: int
    verification_score: dict
    facial_expression_score: dict
    inception_score: dict
    fid: dict


class EvaluateImagePipeline(object):
    def __init__(self, image_dir=None, prompt_type=0, config=None):
        if config is None:
            config = OmegaConf.load(Path(image_dir) / 'configs.yml')
        self.config = config
        self.args = config.train_args
        self.prompt_type = prompt_type

        if prompt_type == 0:
            self.result_yml = self.args.image_output_dir / 'results.yml'
        else:
            self.result_yml = self.args.image_output_dir / \
                f'results{prompt_type}.yml'
        if self.result_yml.exists():
            self.results = OmegaConf.load(self.result_yml)
            self.metrics = self.results
        else:
            facial_expression_score = {
                'sad': 0,
                'angry': 0,
                'surprise': 0,
                'fear': 0,
                'happy': 0,
                'disgust': 0,
            }
            inception_score = {
                'no_sks': 0,
                'no_cond': 0,
            }

            verification_score = deepcopy(facial_expression_score)
            fid = deepcopy(inception_score)

            verification_score['no_sks'] = 0
            verification_score['no_cond'] = 0

            self.metrics = EvaluateMetric(image_dir=str(self.args.image_output_dir),
                                          inference_prompt=inference_prompts[prompt_type],
                                          learning_rate=self.args.learning_rate,
                                          max_train_steps=self.args.max_train_steps,
                                          verification_score=verification_score,
                                          facial_expression_score=facial_expression_score,
                                          inception_score=inception_score,
                                          fid=fid
                                          )

    def save(self):
        OmegaConf.save(self.metrics, self.result_yml)

    def load(self):
        self.results = OmegaConf.load(self.result_yml)

    def analyze_face(self, emotion):
        '''
        args: emotion: DeepFace Emotions as follows
                        DEEPFACE_EMOTIONS = [
                            'sad',
                            'angry',
                            'surprise',
                            'fear',
                            'happy',
                            'disgust',
                            # 'neutral' unconditional
                        ]

        '''
        image_path = self.args.image_output_dir / emotion
        images = list(image_path.glob('*png'))

        if len(images) == 0:
            raise ValueError('Images do not exist. Please generate images.')

        corrects = 0
        for img in tqdm(images):
            results = DeepFace.analyze(str(img), actions=(
                'emotion',), enforce_detection=False, prog_bar=False)
            if results['dominant_emotion'] == emotion:
                corrects += 1

        score = corrects / len(images)
        self.metrics.facial_expression_score[emotion] = score
        self.save()
        return score

    def verify_face(self, image_type, instance_dir=None):
        if instance_dir is None:
            instance_dir = self.args.instance_data_root

        instance_dir = Path(instance_dir)
        instance_images = [p for p in instance_dir.glob(
            '*') if re.search('(jpg|png|jpeg)', str(p))]
        if len(instance_images) == 0:
            raise ValueError('Instance image is not found.')
        image_path = self.args.image_output_dir / image_type
        images = list(image_path.glob('*png'))
        if len(images) == 0:
            raise ValueError('Images do not exist. Please generate images.')

        corrects = 0
        for instance_img in tqdm(instance_images):
            for img in images:
                results = DeepFace.verify(str(instance_img), str(
                    img), enforce_detection=False, prog_bar=False)
                if results['verified']:
                    corrects += 1

        score = corrects / (len(images) * len(instance_images))
        self.metrics.verification_score[image_type] = score
        self.save()

        return score

    def calc_is(self, image_type):
        image_path = self.args.image_output_dir / image_type

        image_files = list(image_path.glob('*png'))
        if len(image_files) == 0:
            raise ValueError('Images do not exist. Please generate images.')

        def open_img(x): return Image.open(x).convert('RGB')
        pil_to_tensor = transforms.PILToTensor()

        images = [pil_to_tensor(open_img(file)) for file in image_files]
        images = torch.stack(images)

        inception = InceptionScore()
        inception.update(images)
        kl_mean, kl_std = inception.compute()

        self.metrics.inception_score[image_type] = kl_mean.item(
        ), kl_std.item()
        self.save()

        return kl_mean.item(), kl_std.item()

    def logger(self):
        self.load()
        if not self.results or not self.config:
            ValueError('Config files are not found.')

        flag = False
        output_file = 'result_log.txt' if self.prompt_type == 0 else f'result_log{self.prompt_type}.txt'
        with open(self.args.image_output_dir / output_file, 'w') as f:
            print('----'*12)
            f.write('----'*12+'\n')
            for k, v in self.config.train_args.items():
                if k in target_keys:
                    if type(v) == omegaconf.dictconfig.DictConfig or type(v) == dict:
                        print(k)
                        f.write(f'{k}\n')
                        for k, v in v.items():
                            print(f'\t{k}: {v}')
                            f.write(f'\t{k}: {v}\n')
                    else:
                        print(f'{k}: {v}')
                        f.write(f'{k}: {v}\n')
                    print('----'*12)
                    f.write('----'*12+'\n')

            print('----'*12)
            f.write('----'*12+'\n')
            for k, v in self.results.items():
                if k in target_keys and not (k == 'learning_rate' or k == 'max_train_steps'):
                    if type(v) == omegaconf.dictconfig.DictConfig or type(v) == dict:
                        print(k)
                        f.write(f'{k}\n')
                        if 'facial_expression_score' in k or 'verification_score' in k:
                            flag = True
                            score = []
                        for k, v in v.items():
                            print(f'\t{k}: {v}')
                            f.write(f'\t{k}: {v}\n')
                            if flag:
                                score.append(v)
                        if flag:
                            avg_score = sum(score) / len(score)
                            print(avg_score)
                            f.write(f'\tAvg Score: {avg_score}\n')
                            flag = False
                    else:
                        print(f'{k}: {v}')
                        f.write(f'{k}: {v}\n')
                    print('----'*12)
                    f.write('----'*12+'\n')
            for k, v in self.config.adapter_config.items():
                if k in target_keys:
                    if type(v) == omegaconf.dictconfig.DictConfig or type(v) == dict:
                        print(k)
                        f.write(f'{k}\n')
                        for k, v in v.items():
                            print(f'\t{k}: {v}')
                            f.write(f'\t{k}: {v}\n')
                    else:
                        print(f'{k}: {v}')
                        f.write(f'{k}: {v}\n')

                    print('----'*12)
                    f.write('----'*12+'\n')


class MyPipeline(DiffusionPipeline):
    r"""
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        new_config = dict(scheduler.config)
        new_config["steps_offset"] = 1
        scheduler._internal_dict = FrozenDict(new_config)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(
                callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(
                text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:,
                                            : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(
            num_images_per_prompt, dim=0)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    " {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(
                batch_size * num_images_per_prompt, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size * num_images_per_prompt,
                         self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not exist on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(
                    latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t,
                                   encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return image
