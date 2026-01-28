# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
hhytuner_dir = os.path.dirname(script_dir)
diffusers_dir = os.path.join(hhytuner_dir, "hhytuner_diffusers")
if diffusers_dir not in sys.path:
    sys.path.insert(0, diffusers_dir)

from diffusers import ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from safetensors.torch import load_file

base_model_path = r"F:\models\Z-Image-Turbo"
lokr_path = r""

output_path = os.path.join(script_dir, "zimage.png")

prompt = '夜景街头特写镜头，中近景构图，人物头部与肩部占据画面主要视觉区域，镜头轻微俯角拍摄，营造出自然的透视纵深感。人物为年轻女性，面部五官精致立体，双眼皮清晰，瞳孔呈深棕色，目光直视镜头，表情略带沉静与微醺感，唇部涂抹暖调红棕色口红，妆容柔和自然，发色为深黑，齐刘海自然垂落，发丝轻柔散落在额前与肩头，皮肤白皙细腻，脸颊带轻微红晕。身体姿态为自然放松的半侧身，颈部与锁骨线条清晰可见，肩部线条流畅，上身穿着深红丝绒质感的吊带装，肩带细窄，一侧肩带呈现豹纹图案，另一侧为纯色丝绒，胸前肌肤裸露，佩戴一串珍珠项链，项链由大小不一的圆润珍珠串联，吊坠为银色金属环扣设计，自然垂落于锁骨下方。背景为城市夜景虚化光斑，呈现暖黄色与冷蓝色交错的圆形光晕，光影柔和，整体氛围朦胧梦幻，色调偏暖，带有轻微胶片颗粒感，画面焦点精准落在人物面部，背景虚化处理强化主体突出。'
negative_prompt = ""
height = 1024
width = 1024
seed = 42

num_inference_steps_base = 9
guidance_scale_base = 0.0

num_inference_steps_lokr = 50
guidance_scale_lokr = 5.0


def detect_lokr_weights(lokr_dir: str) -> bool:
    lokr_weights_path = os.path.join(lokr_dir, "lokr_weights.safetensors")
    lokr_config_path = os.path.join(lokr_dir, "lokr_config.json")
    return os.path.exists(lokr_weights_path) and os.path.exists(lokr_config_path)


def load_lokr_config(lokr_dir: str) -> dict:
    config_path = os.path.join(lokr_dir, "lokr_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_lokr_weights(transformer, lokr_dir: str, device="cuda", dtype=torch.bfloat16):
    from lokr import apply_lokr_to_transformer
    
    lokr_config = load_lokr_config(lokr_dir)
    print(f"[LoKr] 配置: dim={lokr_config['lora_dim']}, alpha={lokr_config['lora_alpha']}, "
          f"factor={lokr_config['lokr_factor']}, modules={lokr_config['num_modules']}")
    
    lokr_modules, lokr_module_names = apply_lokr_to_transformer(
        transformer,
        lora_dim=lokr_config['lora_dim'],
        alpha=lokr_config['lora_alpha'],
        factor=lokr_config['lokr_factor'],
        full_matrix=lokr_config.get('full_matrix', False),
        decompose_both=lokr_config.get('decompose_both', False),
        verbose=True
    )
    
    lokr_weights_path = os.path.join(lokr_dir, "lokr_weights.safetensors")
    print(f"[LoKr] 加载权重: {lokr_weights_path}")
    lokr_state_dict = load_file(lokr_weights_path)
    
    loaded_count = 0
    for idx, (module, layer_name) in enumerate(zip(lokr_modules, lokr_module_names)):
        key_prefix = f"diffusion_model.{layer_name}"
        
        alpha_key = f"{key_prefix}.alpha"
        if alpha_key in lokr_state_dict:
            module.alpha = lokr_state_dict[alpha_key].to(device=device, dtype=dtype)
            module.scale = module.alpha.item() / module.lora_dim
        
        for param_name, param in module.named_parameters():
            full_key = f"{key_prefix}.{param_name}"
            if full_key in lokr_state_dict:
                param.data.copy_(lokr_state_dict[full_key].to(device=device, dtype=dtype))
                loaded_count += 1
    
    print(f"[LoKr] 成功加载 {loaded_count} 个参数")
    
    for module in lokr_modules:
        module.to(device=device, dtype=dtype)
    
    return transformer, lokr_modules


def main():
    name, ext = os.path.splitext(output_path)
    output_path_base = f"{name}_base{ext}"
    output_path_lokr = f"{name}_lokr{ext}"
    
    print("=" * 60)
    print("ZImage LoKr 推理")
    print("=" * 60)
    print(f"提示词: {prompt[:80]}...")
    print(f"负面提示词: {negative_prompt if negative_prompt else '(无)'}")
    print(f"通用参数: height={height}, width={width}, seed={seed}")
    print(f"基础模型: steps={num_inference_steps_base}, guidance_scale={guidance_scale_base}")
    print(f"LoKr 模型: steps={num_inference_steps_lokr}, guidance_scale={guidance_scale_lokr}")
    print("-" * 60)
    
    print(f"[1/2] 加载基础 Pipeline: {base_model_path}")
    pipe = ZImagePipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    
    print("[推理] 基础模型开始生成...")
    start_time = time.time()
    image_base = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps_base,
        guidance_scale=guidance_scale_base,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    end_time = time.time()
    print(f"[推理] 基础模型完成！耗时: {end_time - start_time:.2f} 秒")
    
    image_base.save(output_path_base)
    print(f"[保存] 基础模型图像: {output_path_base}")
    
    print("-" * 60)
    
    image_lokr = None
    
    if lokr_path and os.path.exists(lokr_path) and detect_lokr_weights(lokr_path):
        print(f"[2/2] 检测到 LoKr 权重: {lokr_path}")
        
        print("[加载] 基础 Transformer...")
        base_transformer = ZImageTransformer2DModel.from_pretrained(
            os.path.join(base_model_path, "transformer"),
            torch_dtype=torch.bfloat16,
        )
        base_transformer.to("cuda")
        base_transformer.eval()
        
        lokr_transformer, lokr_modules = apply_lokr_weights(
            base_transformer,
            lokr_path,
            device="cuda",
            dtype=torch.bfloat16,
        )
        
        pipe.transformer = lokr_transformer
        
        print("[推理] LoKr 模型开始生成...")
        start_time = time.time()
        image_lokr = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps_lokr,
            guidance_scale=guidance_scale_lokr,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        end_time = time.time()
        print(f"[推理] LoKr 模型完成！耗时: {end_time - start_time:.2f} 秒")
        
        image_lokr.save(output_path_lokr)
        print(f"[保存] LoKr 模型图像: {output_path_lokr}")
        
        del lokr_transformer
        del lokr_modules
        torch.cuda.empty_cache()
    else:
        print(f"[提示] 未配置 LoKr 路径或路径无效，仅生成基础模型图像")
        print(f"       请设置 lokr_path 变量指向包含 lokr_weights.safetensors 和 lokr_config.json 的目录")
    
    print("=" * 60)
    print("完成！")
    print(f"  基础模型输出: {output_path_base}")
    if image_lokr is not None:
        print(f"  LoKr 模型输出: {output_path_lokr}")
    print("=" * 60)


if __name__ == "__main__":
    main()
