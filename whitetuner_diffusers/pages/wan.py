import os
import sys
import json
import gradio as gr

import gui_common as common

PAGE_TITLE = "WAN I2V LoKr 训练"


def load_gui_config(checkpoint_path):
    if not checkpoint_path or not checkpoint_path.strip():
        return [gr.update()] * 12
    
    checkpoint_path = checkpoint_path.strip()
    config_path = os.path.join(checkpoint_path, "training_config.json")
    
    if not os.path.exists(config_path):
        print(f"未找到配置文件: {config_path}")
        return [gr.update()] * 12
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        print(f"已加载配置: {config_path}")
        
        return [
            gr.update(value=config.get("dit_path", "")),
            gr.update(value=config.get("vae_path", "")),
            gr.update(value=config.get("t5_path", "")),
            gr.update(value=config.get("clip_path", "")),
            gr.update(value=config.get("video_folder", "")),
            gr.update(value=config.get("output_dir", "")),
            gr.update(value=config.get("num_train_steps", 5000)),
            gr.update(value=config.get("learning_rate", 1e-5)),
            gr.update(value=config.get("resolution", 480)),
            gr.update(value=config.get("num_frames", 17)),
            gr.update(value=config.get("timestep_type", "shift")),
            gr.update(value=config.get("shift_scale", 5.0)),
        ]
    except Exception as e:
        print(f"加载配置失败: {e}")
        return [gr.update()] * 12


def start_training(
    dit_path,
    dit_high_noise_path,
    vae_path,
    t5_path,
    clip_path,
    video_folder,
    output_dir,
    wan_version,
    timestep_boundary,
    fp8_scaled,
    blocks_to_swap,
    gradient_checkpointing_cpu_offload,
    use_pinned_memory,
    num_train_steps,
    learning_rate,
    resolution,
    num_frames,
    timestep_type,
    sigmoid_scale,
    shift_scale,
    lognorm_alpha,
    use_caption,
    default_caption,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    resume_from_checkpoint,
):
    has_low = dit_path and dit_path.strip()
    has_high = dit_high_noise_path and dit_high_noise_path.strip()
    
    if not has_low and not has_high:
        yield "[X] 请至少填写一个 DiT 模型路径（Low-noise 或 High-noise）"
        return
    
    if not vae_path:
        yield "[X] 请填写 VAE 模型路径"
        return
    
    if not t5_path:
        yield "[X] 请填写 T5 模型路径"
        return
    
    use_clip = wan_version == "2.1"
    if use_clip and not clip_path:
        yield "[X] WAN 2.1 模式需要 CLIP 模型路径"
        return
    
    if not video_folder:
        yield "[X] 请填写视频数据文件夹路径"
        return
    
    if not os.path.exists(dit_path):
        yield f"[X] DiT 路径不存在: {dit_path}"
        return
    
    if not os.path.exists(vae_path):
        yield f"[X] VAE 路径不存在: {vae_path}"
        return
    
    if not os.path.exists(t5_path):
        yield f"[X] T5 路径不存在: {t5_path}"
        return
    
    if use_clip and clip_path and not os.path.exists(clip_path):
        yield f"[X] CLIP 路径不存在: {clip_path}"
        return
    
    if not os.path.exists(video_folder):
        yield f"[X] 视频文件夹不存在: {video_folder}"
        return
    
    if not output_dir or not output_dir.strip():
        output_dir = os.path.join(common.SCRIPT_DIR, "output")
    
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")
    common.start_tensorboard(logdir=tensorboard_logdir, force_restart=True)
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "wan_trainer.py")
    
    high_low_training = has_low and has_high
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--vae_path", vae_path,
        "--t5_path", t5_path,
        "--video_folder", video_folder,
        "--output_dir", output_dir,
        "--wan_version", wan_version,
        "--num_train_steps", str(int(num_train_steps)),
        "--learning_rate", str(float(learning_rate)),
        "--resolution", str(int(resolution)),
        "--num_frames", str(int(num_frames)),
        "--timestep_type", timestep_type,
        "--sigmoid_scale", str(float(sigmoid_scale)),
        "--shift_scale", str(float(shift_scale)),
        "--lognorm_alpha", str(float(lognorm_alpha)),
        "--checkpoint_every_n_steps", str(int(checkpoint_every_n_steps)),
        "--checkpoints_total_limit", str(int(checkpoints_total_limit)),
    ]
    
    if has_low:
        cmd.extend(["--dit_path", dit_path.strip()])
    
    if has_high:
        cmd.extend(["--dit_high_noise_path", dit_high_noise_path.strip()])
    
    if high_low_training and timestep_boundary and timestep_boundary > 0:
        cmd.extend(["--timestep_boundary", str(float(timestep_boundary))])
    
    if use_clip and clip_path:
        cmd.extend(["--clip_path", clip_path])
    
    if use_caption:
        cmd.append("--use_caption")
    if default_caption:
        cmd.extend(["--default_caption", default_caption])
    
    if fp8_scaled:
        cmd.append("--fp8_scaled")
    
    if blocks_to_swap and int(blocks_to_swap) > 0:
        cmd.extend(["--blocks_to_swap", str(int(blocks_to_swap))])
    
    if gradient_checkpointing_cpu_offload:
        cmd.append("--gradient_checkpointing_cpu_offload")
    
    if use_pinned_memory:
        cmd.append("--use_pinned_memory")
    
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
    
    timestep_info = f"- 时间步采样: {timestep_type}"
    if timestep_type == "sigmoid":
        timestep_info += f" (scale={sigmoid_scale})"
    elif timestep_type == "shift":
        timestep_info += f" (scale={shift_scale})"
    elif timestep_type == "lognorm_blend":
        timestep_info += f" (alpha={lognorm_alpha})"
    
    resume_info = ""
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        resume_info = f"\n- 从检查点恢复: {resume_from_checkpoint}"
    
    clip_info = f"\n- CLIP 模型: {clip_path}" if use_clip else "\n- CLIP: 不需要 (WAN 2.2 模式)"
    
    model_info = ""
    if high_low_training:
        boundary_val = timestep_boundary if timestep_boundary and timestep_boundary > 0 else 0.9
        model_info = f"""
- 训练模式: 双模型训练
- Low-noise DiT: {dit_path}
- High-noise DiT: {dit_high_noise_path}
- Timestep boundary: {boundary_val} (HIGH >= {boundary_val}, LOW < {boundary_val})"""
    elif has_low:
        model_info = f"""
- 训练模式: 单模型 (Low-noise)
- DiT 模型: {dit_path}"""
    else:
        model_info = f"""
- 训练模式: 单模型 (High-noise)
- DiT 模型: {dit_high_noise_path}"""
    
    memory_opts = []
    if fp8_scaled:
        memory_opts.append("FP8 Scaled")
    if blocks_to_swap and int(blocks_to_swap) > 0:
        memory_opts.append(f"Block Swap: {int(blocks_to_swap)}")
    if gradient_checkpointing_cpu_offload:
        memory_opts.append("Activation CPU Offload")
    if use_pinned_memory:
        memory_opts.append("Pinned Memory")
    memory_info = ", ".join(memory_opts) if memory_opts else "None"
    
    initial_msg = f"""使用 accelerate launch 启动 WAN {wan_version} I2V 训练!

配置信息:
- WAN 版本: {wan_version}{model_info}
- VAE 模型: {vae_path}
- T5 模型: {t5_path}{clip_info}
- 视频文件夹: {video_folder}
- 输出目录: {output_dir}
- 训练步数: {num_train_steps}
- 学习率: {learning_rate}
- 分辨率: {resolution}
- 帧数: {num_frames}
{timestep_info}
- 内存优化: {memory_info}
- 使用 Caption: {use_caption}{resume_info}

启动命令: accelerate launch wan_trainer.py ...

正在启动训练流程...

"""
    
    for output in common.run_training_process(cmd, initial_msg=initial_msg):
        yield output


def create_page():
    with gr.Column() as page:
        gr.Markdown(
            """
            ### WAN I2V LoKr 训练
            
            WAN (Wan-Video) 图生视频模型的 LoKr 微调训练。
            - 使用 LoKr (Low-Rank Kronecker) 高效微调，显存占用更低
            - 支持 I2V (Image to Video) 模式
            - 支持视频文件或图片序列文件夹作为训练数据
            - 支持 caption 文本描述（放在视频同名 .txt 文件中）
            - 输出格式兼容 LyCORIS，可在 ComfyUI 中直接加载
            
            **[致谢]** 本训练器的大部分代码来自 kohya-ss/musubi-tuner 项目
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**模型配置**")
                
                wan_version = gr.Radio(
                    label="WAN 版本",
                    choices=[
                        ("WAN 2.2 (推荐，不需要 CLIP)", "2.2"),
                        ("WAN 2.1 (需要 CLIP)", "2.1"),
                    ],
                    value="2.2",
                    info="WAN 2.2 不需要 CLIP 模型，只需要 T5"
                )
                
                gr.Markdown("**模型路径** (至少填一个，填两个则为双模型训练)")
                
                with gr.Row():
                    wan_dit_path = gr.Textbox(
                        label="Low-noise DiT 路径",
                        placeholder="选择 low_noise_model 或单模型 wan2.x_i2v.safetensors",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_dit_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                wan_high_noise_row = gr.Row(visible=True)
                with wan_high_noise_row:
                    wan_dit_high_noise_path = gr.Textbox(
                        label="High-noise DiT 路径 (可选，填写则启用双模型训练)",
                        placeholder="选择 high_noise_model，或留空只训练 Low-noise",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_dit_high_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    wan_vae_path = gr.Textbox(
                        label="VAE 模型路径",
                        placeholder="选择 wan_2.1_vae.safetensors 或 Wan2.1_VAE.pth",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_vae_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    wan_t5_path = gr.Textbox(
                        label="T5 模型路径",
                        placeholder="选择 models_t5_umt5-xxl-enc-bf16.pth",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_t5_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                wan_clip_row = gr.Row(visible=False)
                with wan_clip_row:
                    wan_clip_path = gr.Textbox(
                        label="CLIP 模型路径 (仅 WAN 2.1 需要)",
                        placeholder="选择 models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_clip_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                gr.Markdown("**数据路径**")
                
                with gr.Row():
                    wan_video_folder = gr.Textbox(
                        label="视频数据文件夹",
                        placeholder="包含视频文件(.mp4)或图片序列子文件夹",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_video_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    wan_output_dir = gr.Textbox(
                        label="输出目录",
                        placeholder="选择模型输出保存目录",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_output_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    wan_resume_checkpoint = gr.Textbox(
                        label="从检查点恢复 (可选)",
                        placeholder="选择 checkpoint-xxx 文件夹以继续训练",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    wan_resume_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                wan_use_caption = gr.Checkbox(
                    label="使用 Caption（读取同名 .txt 文件）",
                    value=True,
                )
                
                wan_default_caption = gr.Textbox(
                    label="默认 Caption（无 .txt 文件时使用）",
                    value="",
                    placeholder="可选：为没有 caption 的视频设置默认描述",
                )
                
                with gr.Accordion("高级参数", open=False):
                    gr.Markdown("**内存优化选项**")
                    
                    wan_fp8_scaled = gr.Checkbox(
                        label="FP8 Scaled 量化",
                        value=True,
                        info="将 DiT 模型权重量化为 FP8 格式，显著减少显存占用（LoKr 训练支持）"
                    )
                    
                    wan_blocks_to_swap = gr.Slider(
                        label="Block Swap 数量",
                        minimum=0,
                        maximum=39,
                        value=0,
                        step=1,
                        info="将部分 transformer blocks 换出到 CPU，减少显存占用但会降低训练速度。0 表示禁用"
                    )
                    
                    wan_gradient_checkpointing_cpu_offload = gr.Checkbox(
                        label="Activation CPU Offloading",
                        value=False,
                        info="将激活值卸载到 CPU，配合 gradient checkpointing 进一步减少显存"
                    )
                    
                    wan_use_pinned_memory = gr.Checkbox(
                        label="Use Pinned Memory",
                        value=False,
                        info="使用固定内存加速 Block Swap 的数据传输"
                    )
                    
                    gr.Markdown("**训练参数**")
                    
                    wan_num_train_steps = gr.Number(
                        label="训练步数",
                        value=5000,
                        info="总训练步数"
                    )
                    
                    wan_learning_rate = gr.Number(
                        label="学习率",
                        value=1e-4,
                        info="LoKr 训练建议 1e-4"
                    )
                    
                    wan_resolution = gr.Number(
                        label="视频分辨率",
                        value=480,
                        info="视频短边分辨率（如 480 表示 480p）"
                    )
                    
                    wan_num_frames = gr.Number(
                        label="视频帧数",
                        value=17,
                        info="每个视频的训练帧数"
                    )
                    
                    gr.Markdown("**时间步采样设置**")
                    
                    wan_timestep_type = gr.Dropdown(
                        label="时间步采样类型",
                        choices=[
                            ("linear - 均匀分布，通用场景", "linear"),
                            ("sigmoid - 集中中间，适合细节和风格", "sigmoid"),
                            ("weighted - 中间权重高，适合蒸馏模型", "weighted"),
                            ("shift - 偏向高噪声，推荐用于视频", "shift"),
                            ("lognorm_blend - 混合分布，平衡构图和细节", "lognorm_blend"),
                        ],
                        value="shift",
                        info="WAN 官方推荐使用 shift 采样"
                    )
                    
                    wan_sigmoid_scale = gr.Slider(
                        label="Sigmoid Scale (仅 sigmoid 有效)",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        info="分布集中程度"
                    )
                    
                    wan_shift_scale = gr.Slider(
                        label="Shift Scale (仅 shift 有效)",
                        minimum=1.0,
                        maximum=10.0,
                        value=5.0,
                        step=0.5,
                        info="WAN 官方推荐值为 5.0"
                    )
                    
                    wan_lognorm_alpha = gr.Slider(
                        label="LogNorm Alpha (仅 lognorm_blend 有效)",
                        minimum=0.5,
                        maximum=0.9,
                        value=0.75,
                        step=0.05,
                        info="对数正态分布比例"
                    )
                    
                    gr.Markdown("**WAN 2.2 双模型设置**")
                    
                    wan_timestep_boundary = gr.Slider(
                        label="Timestep Boundary (双模型分界)",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        info="I2V 默认 0.9 (HIGH >= 0.9, LOW < 0.9)"
                    )
                    
                    gr.Markdown("**检查点设置**")
                    
                    wan_checkpoint_every = gr.Number(
                        label="检查点保存间隔",
                        value=500,
                        info="每多少步保存一次"
                    )
                    
                    wan_checkpoints_limit = gr.Number(
                        label="检查点保留数量",
                        value=3,
                        info="最多保留多少个"
                    )
                
                def on_version_change(version):
                    show_clip = version == "2.1"
                    show_high_noise = version == "2.2"
                    return gr.update(visible=show_clip), gr.update(visible=show_high_noise)
                
                wan_version.change(
                    fn=on_version_change,
                    inputs=wan_version,
                    outputs=[wan_clip_row, wan_high_noise_row]
                )
                
                if common.is_local_mode:
                    wan_dit_btn.click(fn=common.select_file, inputs=wan_dit_path, outputs=wan_dit_path)
                    wan_dit_high_btn.click(fn=common.select_file, inputs=wan_dit_high_noise_path, outputs=wan_dit_high_noise_path)
                    wan_vae_btn.click(fn=common.select_file, inputs=wan_vae_path, outputs=wan_vae_path)
                    wan_t5_btn.click(fn=common.select_file, inputs=wan_t5_path, outputs=wan_t5_path)
                    wan_clip_btn.click(fn=common.select_file, inputs=wan_clip_path, outputs=wan_clip_path)
                    wan_video_btn.click(fn=common.select_folder, inputs=wan_video_folder, outputs=wan_video_folder)
                    wan_output_btn.click(fn=common.select_folder, inputs=wan_output_dir, outputs=wan_output_dir)
            
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        wan_start_btn = gr.Button(
                            "开始 WAN I2V LoKr 训练",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        wan_stop_btn = gr.Button(
                            "停止训练",
                            variant="stop",
                            size="lg",
                            scale=1
                        )
                    
                    wan_status_text = gr.Textbox(
                        label="训练状态",
                        value="等待启动...\n\n配置好参数后点击上方按钮开始 WAN I2V LoKr 训练\n\n数据格式说明:\n- 视频文件: 直接放 .mp4/.avi/.mov 文件\n- 图片序列: 每个子文件夹为一个视频\n- Caption: 与视频同名的 .txt 文件\n\nLoKr 训练特点:\n- 显存占用低，适合消费级显卡\n- 输出 LyCORIS 格式，可在 ComfyUI 加载",
                        interactive=False,
                        lines=28
                    )
                
                config_outputs = [
                    wan_dit_path,
                    wan_vae_path,
                    wan_t5_path,
                    wan_clip_path,
                    wan_video_folder,
                    wan_output_dir,
                    wan_num_train_steps,
                    wan_learning_rate,
                    wan_resolution,
                    wan_num_frames,
                    wan_timestep_type,
                    wan_shift_scale,
                ]
                
                wan_start_btn.click(
                    fn=load_gui_config,
                    inputs=wan_resume_checkpoint,
                    outputs=config_outputs
                ).then(
                    fn=start_training,
                    inputs=[
                        wan_dit_path,
                        wan_dit_high_noise_path,
                        wan_vae_path,
                        wan_t5_path,
                        wan_clip_path,
                        wan_video_folder,
                        wan_output_dir,
                        wan_version,
                        wan_timestep_boundary,
                        wan_fp8_scaled,
                        wan_blocks_to_swap,
                        wan_gradient_checkpointing_cpu_offload,
                        wan_use_pinned_memory,
                        wan_num_train_steps,
                        wan_learning_rate,
                        wan_resolution,
                        wan_num_frames,
                        wan_timestep_type,
                        wan_sigmoid_scale,
                        wan_shift_scale,
                        wan_lognorm_alpha,
                        wan_use_caption,
                        wan_default_caption,
                        wan_checkpoint_every,
                        wan_checkpoints_limit,
                        wan_resume_checkpoint,
                    ],
                    outputs=wan_status_text,
                    show_progress="full"
                )
                
                wan_stop_btn.click(
                    fn=common.stop_training,
                    inputs=None,
                    outputs=None
                )
                
                if common.is_local_mode:
                    wan_resume_btn.click(
                        fn=common.select_folder,
                        inputs=wan_resume_checkpoint,
                        outputs=wan_resume_checkpoint
                    ).then(
                        fn=load_gui_config,
                        inputs=wan_resume_checkpoint,
                        outputs=config_outputs
                    )
    
    return page

