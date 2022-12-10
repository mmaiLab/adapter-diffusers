import math
import os, sys, datetime
import random
import wandb
from typing import Optional
from pathlib import Path
import argparse

import numpy as np
import torch, torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.tracking import GeneralTracker
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler, 
    PNDMScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
#sys.path.append(os.environ['LAB'] + 'diffusion/')
from modeling.ada_unet import AdaUNet2DConditionModel
from modeling.ada_clip import AdaCLIPTextModel
from modeling.ada_vae import AdaAutoencoderKL
from modeling.adapter import AdapterConfig
from modeling.pipeline import MyPipeline
from modeling.utils import (
    DreamBoothDataset, 
    ActionPromptDataset,
    TemplateDataset,
    collate_fn,
    PromptDataset,
    image_grid, 
    save_adapter, 
    load_adapter, 
    get_template,
    TEMPLATES,
    load_base_model,
    TrainArgs,
    ExperimentConfig,
)
from modeling.utils import freeze_params, show_params_grad
from omegaconf import OmegaConf
from distutils.util import strtobool

ROOT = Path('/gs/hs1/tga-i/otake.s.ad/diffusion')

SCHEDULERS = {
    'ddpm': DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000),
    'ddim': DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000),
    'pndm': PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000),
    'lmsd': LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
}



def training_function(models, device, args, use_wandb, model_save_name=None):
    
    tokenizer = models['tokenizer']
    text_encoder = models['text_encoder']
    unet = models['unet']
    vae = models['vae']
    
    param_to_update = []
    if args.on_finetuning:
        param_to_update += [param for param in unet.parameters()]
        param_to_update += [param for param in vae.parameters()]    
    else:
        param_to_update += freeze_params(unet)
        param_to_update += freeze_params(vae)
        param_to_update += freeze_params(text_encoder)
    
    if use_wandb:
        wandb.init(
            project='StableDiffusion',
            id=args.run_name,
            config=args
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    accelerator.state.device = device

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_vae:
            vae.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    
    if args.style_name:
        optimizer_instance = torch.optim.AdamW(
            param_instance,
            lr=args.learning_rate['instance'],
        )
        optimizer_style = torch.optim.AdamW(
            param_style,
            lr=args.learning_rate['style']
        )
        
    else:
        optimizer = torch.optim.AdamW(
            param_to_update,
            lr=args.learning_rate,
        )
    
    noise_scheduler = SCHEDULERS[args.noise_scheduler['train']]
    
    train_dataset = TemplateDataset(
        class_name=args.class_name,
        instance_data_root=args.instance_data_root,
        instance_prompt=args.instance_prompt,
        instance_data_ratio=args.instance_data_ratio,
        style_name=args.style_name,
        style_data_root=args.style_data_root,
        style_templates=args.style_templates,
        style_data_ratio=args.style_data_ratio,
        tokenizer=tokenizer,
    )
    
    train_dataloader = train_dataset.create_dataloader(args.train_batch_size)
    
    
    if args.train_text_encoder and args.train_vae:
        unet, text_encoder, vae, optimizer, train_dataloader = accelerator.prepare(unet, text_encoder, vae, optimizer, train_dataloader)
    elif args.train_vae:
        unet, vae, optimizer, train_dataloader = accelerator.prepare(unet, vae, optimizer, train_dataloader)
        text_encoder.to(accelerator.device)
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
        text_encoder.to(accelerator.device)
        vae.to(accelerator.device)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
  
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        if args.train_vae:
            vae.train()
            
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with accelerator.accumulate(vae):
                    with accelerator.accumulate(text_encoder):
                        # Convert images to latent space
                        sample_image = batch["pixel_values"]
                        if args.train_vae == 'on_rec':
                            latents_vae = vae.encode(sample_image).latent_dist.sample()
                            latents_vae = latents_vae * 0.18215
                            latents = latents_vae.clone().detach()
                        elif args.train_vae == 'on_ldm':
                            latents = vae.encode(sample_image).latent_dist.sample()
                            latents = latents * 0.18215
                            latents_vae = latents.clone().detach()
                        else:    
                            with torch.no_grad():
                                latents = vae.encode(sample_image).latent_dist.sample()
                                latents = latents * 0.18215
                                latents_vae = latents.clone()
                        
                        latents.requires_grad=True
                        
                        # Sample noise that we'll add to the latents
                        noise = torch.randn(latents.shape).to(latents.device)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                        ).long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        if args.train_text_encoder:
                            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                            encoder_hidden_states.requires_grad = True
                        else:
                            with torch.no_grad():
                                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        logs = {}

                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                        loss.backward()
                        #del loss
                        torch.cuda.empty_cache()

                        if args.train_vae:
                            latents_vae = 1 / 0.18215 * latents_vae
                            reconst_image = vae.decode(latents_vae).sample
                            loss_vae = F.mse_loss(reconst_image, sample_image, reduction='none').mean([1, 2, 3]).mean()
                            loss_vae.backward()
                            #del loss_vae
                            torch.cuda.empty_cache()
                        #loss = loss+ loss_vae
                        #loss.backward()
                        torch.cuda.empty_cache()

                        if args.max_grad_norm:
                            accelerator.clip_grad_norm_(param_to_update, args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs.update({"loss": loss.detach().item()})
            
            if args.train_vae:
                logs.update({'loss_vae': loss_vae.detach().item()})
            
            progress_bar.set_postfix(**logs)
            
            if accelerator.is_main_process:
                if use_wandb:
                    wandb.log(logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        output_dir = args.output_dir
        if 'style' in args.run_name:
            if model_save_name:
                save_adapter(accelerator.unwrap_model(unet), output_dir / model_save_name)
            else:    
                save_adapter(accelerator.unwrap_model(unet), output_dir / 'unet_adapter_style.pth')
        else:
            save_adapter(accelerator.unwrap_model(unet), output_dir / 'unet_adapter.pth')
        
        if args.train_text_encoder:
            save_adapter(accelerator.unwrap_model(text_encoder), output_dir / 'text_adapter.pth')
        
        if args.train_vae:
            save_adapter(accelerator.unwrap_model(vae), output_dir / 'vae_adapter.pth')

def init_config():
    args = TrainArgs(instance_data_root='vgg_face2-master/samples/loose_crop (release version)/n000106')
    adapter_config = AdapterConfig()
    config = ExperimentConfig(train_args=args, adapter_config=adapter_config)
    OmegaConf.save(config, ROOT / 'init_config' / 'configs.yml')
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_file', nargs='*', type=str, default=None)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--init_config', type=bool, default=False)
    parser.add_argument('--log_wandb', type=strtobool, default=True)
    parser.add_argument('--identifier', type=str, default=None)
    parser.add_argument('--skip_instance', type=strtobool, default=False)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--model_save_name', type=str, default=None)
    
    
    args = parser.parse_args()
    
    if args.init_config:
        init_config()
        sys.exit(0)
        
    config_files = args.config_file
    device = f'cuda:{args.device_num}'
    
    if config_files is not None:
        if len(config_files) > 1: # train both instance and style adpaters
            config = OmegaConf.load(config_files[0]) # for instance
            config_style = OmegaConf.load(config_files[1]) # for style
        else: # train only style adapter
            config_file = config_files[0]
            config = OmegaConf.load(config_file)
    else: # if config file is not specified, training is done based on an init config 
        init_config_path = ROOT / 'init_config'
        config = OmegaConf.load(init_config_path / f'configs_{args.device_num}.yml')

    train_args = config.train_args
    adapter_config = config.adapter_config
    
    if train_args.on_finetuning: # fine-tuning
        models = load_base_model(adapter_config, 
                         use_ada_unet=False, 
                         use_ada_vae=False, 
                         use_ada_text_encoder=False)
    else: # adapter-tuning
        models = load_base_model(adapter_config, 
                                 use_ada_unet=True, 
                                 use_ada_vae=True if train_args.train_vae else False, 
                                 use_ada_text_encoder=False)
    
    if not args.identifier:
        identifier = train_args.instance_data_root[-7:]
    else:
        identifier = args.identifier
    
    if args.skip_instance:
        model_save_name = args.model_save_name
        model_path = ROOT / 'model'
        timestamp = args.timestamp
        train_args.output_dir = next(model_path.glob(f'*{timestamp}*')) # specify a directory based on timestamp
        config_save_name = f'configs_{model_save_name}.yml' if model_save_name else 'configs_style.yml'
    else:
        model_save_name = None
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        config_save_name = 'configs.yml'
    
    if train_args.on_finetuning:
        train_args.run_name=f'finetuning_vggface_{identifier}_lr-{train_args.learning_rate}_steps-{train_args.max_train_steps}_{timestamp}'
    else:
        if args.skip_instance:
            train_args.run_name=f'style_train_vae-{train_args.train_vae}_lr-{train_args.learning_rate}_{timestamp}'
        else:
            train_args.run_name=f'{identifier}_train_vae-{train_args.train_vae}_lr-{train_args.learning_rate}_{timestamp}'


    if len(config_files) > 1: # if train both instance and style adapters
        # store info
        train_args.output_dir = Path(train_args.output_dir) / train_args.run_name
        output_dir = train_args.output_dir
        image_output_dir = train_args.image_output_dir
        model_save_name = args.model_save_name

    train_args.image_output_dir = Path(str(train_args.output_dir).replace('model', 'sample'))
    
    configs = ExperimentConfig(train_args=train_args, adapter_config=adapter_config)

    modules = adapter_config.modules['transformer'] + adapter_config.modules['resnet'] + adapter_config.add_modules
    modules = '_'.join(modules)

    train_args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerate.notebook_launcher(training_function, args=(models, device, train_args, args.log_wandb, model_save_name), num_processes=1)
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    train_args.image_output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(configs, train_args.output_dir / config_save_name)
    OmegaConf.save(configs, train_args.image_output_dir / config_save_name)
    
    if len(config_files) > 1: # if train both instance and style adapters
        train_args = config_style.train_args
        adapter_config = config_style.adapter_config
        train_args.run_name=f'style_train_vae-{train_args.train_vae}_lr-{train_args.learning_rate}_{timestamp}'
        train_args.output_dir = output_dir
        train_args.image_output_dir = image_output_dir
        style_modules = adapter_config.modules['transformer'] + adapter_config.modules['resnet'] + adapter_config.add_modules
        style_modules = '_'.join(style_modules)
        models = load_base_model(adapter_config, 
                         use_ada_unet=True, 
                         use_ada_vae=True if train_args.train_vae else False, 
                         use_ada_text_encoder=False)
        model_save_name = args.model_save_name
        accelerate.notebook_launcher(training_function, args=(models, device, train_args, args.log_wandb, model_save_name), num_processes=1)
        with torch.no_grad():
            torch.cuda.empty_cache()
            
        configs = ExperimentConfig(train_args=train_args, adapter_config=adapter_config)
        OmegaConf.save(configs, train_args.output_dir / 'configs_style.yml')
        OmegaConf.save(configs, train_args.image_output_dir / 'configs_style.yml')
        
        
        with open('logs.txt', 'a') as f:
            f.write(f'{timestamp}\n\tmodules={modules}_stylemodules={style_modules}\n\n')
        
    else:
        # save logs
        if not args.skip_instance:
            with open('logs.txt', 'a') as f:
                f.write(f'{timestamp}\n\tmodules={modules}\n\tlr={train_args.learning_rate}\n\tmaxTrainSteps={train_args.max_train_steps}\n\tinstanceFile=vggface-{identifier}\n\ttrain_vae={train_args.train_vae}\n\n')
        
    
    
    
if __name__=='__main__':
    main()