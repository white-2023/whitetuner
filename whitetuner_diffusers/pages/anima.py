import os
import sys
import gradio as gr

import gui_common as common

PAGE_TITLE = "Anima T2I"


def start_training(
    dit_path,
    vae_path,
    text_encoder_path,
    image_folder,
    output_dir,
    resume_from_checkpoint,
    num_train_steps,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    learning_rate,
    train_text_encoder,
    te_learning_rate,
    batch_size,
    resolution,
    timestep_type,
    sigmoid_scale,
    shift_scale,
    lognorm_alpha,
    use_caption,
    caption_ext,
    default_caption,
    noise_offset,
    blocks_to_swap,
    use_adafactor,
    use_pinned_memory,
):
    if not dit_path:
        yield "[X] 请填写 DiT 模型路径"
        return
    
    if not vae_path:
        yield "[X] 请填写 VAE 模型路径"
        return
    
    if not text_encoder_path:
        yield "[X] 请填写文本编码器路径"
        return
    
    
    if not image_folder:
        yield "[X] 请填写图片文件夹路径"
        return
    
    if not os.path.exists(dit_path):
        yield f"[X] DiT 路径不存在: {dit_path}"
        return
    
    if not os.path.exists(vae_path):
        yield f"[X] VAE 路径不存在: {vae_path}"
        return
    
    if not os.path.exists(text_encoder_path):
        yield f"[X] 文本编码器路径不存在: {text_encoder_path}"
        return
    
    
    if not os.path.exists(image_folder):
        yield f"[X] 图片文件夹不存在: {image_folder}"
        return
    
    if not output_dir or not output_dir.strip():
        output_dir = os.path.join(common.SCRIPT_DIR, "output")
    
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")
    common.start_tensorboard(logdir=tensorboard_logdir, force_restart=True)
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "anima_trainer.py")
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--dit_path", dit_path,
        "--vae_path", vae_path,
        "--text_encoder_path", text_encoder_path,
        "--image_folder", image_folder,
        "--output_dir", output_dir,
        "--num_train_steps", str(int(num_train_steps)),
        "--checkpoint_every_n_steps", str(int(checkpoint_every_n_steps)),
        "--checkpoints_total_limit", str(int(checkpoints_total_limit)),
        "--learning_rate", str(float(learning_rate)),
        "--batch_size", str(int(batch_size)),
        "--resolution", str(int(resolution)),
        "--timestep_type", timestep_type,
        "--sigmoid_scale", str(float(sigmoid_scale)),
        "--shift_scale", str(float(shift_scale)),
        "--lognorm_alpha", str(float(lognorm_alpha)),
        "--noise_offset", str(float(noise_offset)),
    ]
    
    if train_text_encoder:
        cmd.extend(["--train_text_encoder"])
        cmd.extend(["--te_learning_rate", str(float(te_learning_rate))])
    
    if blocks_to_swap and int(blocks_to_swap) > 0:
        cmd.extend(["--blocks_to_swap", str(int(blocks_to_swap))])
    
    if use_adafactor:
        cmd.append("--use_adafactor")
    
    if use_pinned_memory:
        cmd.append("--use_pinned_memory")
    
    if use_caption:
        cmd.append("--use_caption")
        cmd.extend(["--caption_ext", caption_ext])
    
    if default_caption and default_caption.strip():
        cmd.extend(["--default_caption", default_caption.strip()])
    
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
    
    initial_msg = f"""使用 accelerate launch 启动 Anima T2I 训练!

配置信息:
- DiT 模型: {dit_path}
- VAE 模型: {vae_path}
- 文本编码器: {text_encoder_path}
- 图片文件夹: {image_folder}
- 输出目录: {output_dir}
- 训练步数: {num_train_steps}
- 学习率: {learning_rate}
- 分辨率: {resolution}
{timestep_info}
- Noise Offset: {noise_offset}
- 检查点间隔: {checkpoint_every_n_steps}{resume_info}

启动命令: accelerate launch anima_trainer.py ...

正在启动训练流程...

"""
    
    for output in common.run_training_process(cmd, initial_msg=initial_msg):
        yield output


def create_page():
    with gr.Column() as page:
        gr.Markdown(
            """
            ### Anima T2I 全量训练
            
            Anima (circlestone-labs) 文生图模型的全量微调训练。
            - 基于 NVIDIA Cosmos-Predict2-2B 架构
            - 使用 Qwen3 0.6B 作为文本编码器
            - 使用 Wan 2.1 VAE 进行图像编解码
            - 专注于动漫风格生成
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**模型配置**")
                
                default_caption = gr.Textbox(
                    label="默认 Caption",
                    value="",
                    placeholder="anime style, high quality",
                    info="当图片没有对应的 caption 文件时使用此默认值"
                )
                
                with gr.Row():
                    dit_path = gr.Textbox(
                        label="DiT 模型路径",
                        placeholder="选择 anima-preview.safetensors",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    dit_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    vae_path = gr.Textbox(
                        label="VAE 模型路径",
                        placeholder="选择 qwen_image_vae.safetensors",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    vae_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    text_encoder_path = gr.Textbox(
                        label="文本编码器路径",
                        placeholder="选择 qwen_3_06b_base.safetensors",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    te_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                gr.Markdown("**数据路径**")
                
                with gr.Row():
                    image_folder = gr.Textbox(
                        label="图片文件夹",
                        placeholder="选择训练图片文件夹",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    image_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="输出目录",
                        placeholder="选择模型输出保存目录",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    output_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                with gr.Row():
                    resume_from_checkpoint = gr.Textbox(
                        label="从检查点恢复 (可选)",
                        placeholder="选择 checkpoint-xxx 文件夹以继续训练",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    resume_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    dit_btn.click(fn=common.select_file, inputs=dit_path, outputs=dit_path)
                    vae_btn.click(fn=common.select_file, inputs=vae_path, outputs=vae_path)
                    te_btn.click(fn=common.select_file, inputs=text_encoder_path, outputs=text_encoder_path)
                    image_btn.click(fn=common.select_folder, inputs=image_folder, outputs=image_folder)
                    output_btn.click(fn=common.select_folder, inputs=output_dir, outputs=output_dir)
                    resume_btn.click(fn=common.select_folder, inputs=resume_from_checkpoint, outputs=resume_from_checkpoint)
                
                with gr.Accordion("高级参数", open=False):
                    use_caption = gr.Checkbox(
                        label="使用 Caption 文件",
                        value=True,
                        info="读取与图片同名的 .txt 文件作为 caption"
                    )
                    
                    caption_ext = gr.Textbox(
                        label="Caption 文件扩展名",
                        value=".txt",
                        info="Caption 文件的扩展名"
                    )
                    
                    num_train_steps = gr.Number(
                        label="训练步数",
                        value=5000,
                        info="总训练步数"
                    )
                    
                    learning_rate = gr.Number(
                        label="学习率 (DiT)",
                        value=1e-5,
                        info="DiT 全量训练建议 1e-5 ~ 5e-5"
                    )
                    
                    train_text_encoder = gr.Checkbox(
                        label="训练 Text Encoder",
                        value=False,
                        info="同时训练 Qwen3 0.6B Text Encoder (增加约 1-2GB 显存)"
                    )
                    
                    te_learning_rate = gr.Number(
                        label="Text Encoder 学习率",
                        value=1e-6,
                        info="TE 学习率建议比 DiT 低一个数量级",
                        visible=False
                    )
                    
                    train_text_encoder.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[train_text_encoder],
                        outputs=[te_learning_rate]
                    )
                    
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        info="每个 GPU 的批次大小"
                    )
                    
                    resolution = gr.Number(
                        label="分辨率",
                        value=1024,
                        info="训练图片的最大边长"
                    )
                    
                    checkpoint_every_n_steps = gr.Number(
                        label="检查点保存间隔",
                        value=500,
                        info="每多少步保存一次"
                    )
                    
                    checkpoints_total_limit = gr.Number(
                        label="检查点保留数量",
                        value=3,
                        info="最多保留多少个"
                    )
                    
                    gr.Markdown("**时间步采样设置**")
                    
                    timestep_type = gr.Dropdown(
                        label="时间步采样类型",
                        choices=[
                            ("shift - 偏向高噪声，推荐", "shift"),
                            ("sigmoid - 集中中间，适合细节和风格", "sigmoid"),
                            ("linear - 均匀分布，通用场景", "linear"),
                            ("lognorm_blend - 混合分布，平衡构图和细节", "lognorm_blend"),
                        ],
                        value="shift",
                        info="Anima 推荐使用 shift 采样 (scale=3.0)"
                    )
                    
                    sigmoid_scale = gr.Slider(
                        label="Sigmoid Scale (仅 sigmoid 有效)",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        info="分布集中程度，越大越集中在中间"
                    )
                    
                    shift_scale = gr.Slider(
                        label="Shift Scale (仅 shift 有效)",
                        minimum=1.0,
                        maximum=6.0,
                        value=3.0,
                        step=0.5,
                        info="Anima 推荐值为 3.0"
                    )
                    
                    lognorm_alpha = gr.Slider(
                        label="LogNorm Alpha (仅 lognorm_blend 有效)",
                        minimum=0.5,
                        maximum=0.9,
                        value=0.75,
                        step=0.05,
                        info="对数正态分布比例"
                    )
                    
                    gr.Markdown("**正则化设置**")
                    
                    noise_offset = gr.Slider(
                        label="Noise Offset",
                        minimum=0.0,
                        maximum=0.1,
                        value=0.0,
                        step=0.01,
                        info="噪声偏移，可以帮助生成更亮/更暗的图像"
                    )
                    
                    gr.Markdown("**显存优化设置**")
                    
                    blocks_to_swap = gr.Slider(
                        label="Block Swap 数量",
                        minimum=0,
                        maximum=27,
                        value=0,
                        step=1,
                        info="0=禁用，推荐 20 (Anima有28个block)"
                    )
                    
                    use_adafactor = gr.Checkbox(
                        label="使用 Adafactor 优化器",
                        value=False,
                        info="Block Swap 时必须开启",
                        visible=False
                    )
                    
                    use_pinned_memory = gr.Checkbox(
                        label="使用 Pinned Memory",
                        value=False,
                        info="可以加速 CPU-GPU 传输",
                        visible=False
                    )
                    
                    def on_block_swap_change(swap_count):
                        if swap_count > 0:
                            return (
                                gr.update(visible=True, value=True),
                                gr.update(visible=True, value=True),
                            )
                        else:
                            return (
                                gr.update(visible=False, value=False),
                                gr.update(visible=False, value=False),
                            )
                    
                    blocks_to_swap.change(
                        fn=on_block_swap_change,
                        inputs=[blocks_to_swap],
                        outputs=[use_adafactor, use_pinned_memory]
                    )
            
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        start_btn = gr.Button(
                            "开始 Anima 训练",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        stop_btn = gr.Button(
                            "停止训练",
                            variant="stop",
                            size="lg",
                            scale=1
                        )
                    
                    status_text = gr.Textbox(
                        label="训练状态",
                        value="""等待启动...

配置好参数后点击上方按钮开始 Anima T2I 训练

模型来源:
- circlestone-labs/Anima
- 基于 NVIDIA Cosmos-Predict2-2B-Text2Image

数据格式说明:
- 图片: .jpg/.png/.webp 等常见格式
- Caption: 与图片同名的 .txt 文件

推荐配置:
- 分辨率: 1024
- 学习率: 1e-5
- Shift Scale: 3.0""",
                        interactive=False,
                        lines=28
                    )
                
                start_btn.click(
                    fn=start_training,
                    inputs=[
                        dit_path,
                        vae_path,
                        text_encoder_path,
                        image_folder,
                        output_dir,
                        resume_from_checkpoint,
                        num_train_steps,
                        checkpoint_every_n_steps,
                        checkpoints_total_limit,
                        learning_rate,
                        train_text_encoder,
                        te_learning_rate,
                        batch_size,
                        resolution,
                        timestep_type,
                        sigmoid_scale,
                        shift_scale,
                        lognorm_alpha,
                        use_caption,
                        caption_ext,
                        default_caption,
                        noise_offset,
                        blocks_to_swap,
                        use_adafactor,
                        use_pinned_memory,
                    ],
                    outputs=status_text,
                    show_progress="full"
                )
                
                stop_btn.click(
                    fn=common.stop_training,
                    inputs=None,
                    outputs=None
                )
    
    return page
