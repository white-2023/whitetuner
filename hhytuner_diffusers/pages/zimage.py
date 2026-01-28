import os
import sys
import gradio as gr

import gui_common as common

PAGE_TITLE = "ZImage 训练"


def start_training(
    model_id,
    image_folder,
    output_dir,
    resume_from_checkpoint,
    training_mode,
    num_train_steps,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    learning_rate,
    resolution,
    use_caption,
    default_caption,
    timestep_type,
    sigmoid_scale,
    shift_scale,
    lognorm_alpha,
    loss_weighting_scheme,
    prompt_dropout_prob,
    noise_offset,
):
    if not model_id:
        yield "[X] 请填写 ZImage 模型路径"
        return
    
    if not image_folder:
        yield "[X] 请填写训练图片文件夹路径"
        return
    
    if not os.path.exists(model_id):
        yield f"[X] 模型路径不存在: {model_id}"
        return
    
    if not os.path.exists(image_folder):
        yield f"[X] 图片文件夹不存在: {image_folder}"
        return
    
    if not output_dir or not output_dir.strip():
        output_dir = os.path.join(common.SCRIPT_DIR, "output")
    
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")
    common.start_tensorboard(logdir=tensorboard_logdir, force_restart=True)
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "zimage_trainer.py")
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--model_id", model_id,
        "--image_folder", image_folder,
        "--output_dir", output_dir,
        "--training_mode", training_mode,
        "--num_train_steps", str(int(num_train_steps)),
        "--checkpoint_every_n_steps", str(int(checkpoint_every_n_steps)),
        "--checkpoints_total_limit", str(int(checkpoints_total_limit)),
        "--learning_rate", str(float(learning_rate)),
        "--resolution", str(int(resolution)),
        "--timestep_type", timestep_type,
        "--sigmoid_scale", str(float(sigmoid_scale)),
        "--shift_scale", str(float(shift_scale)),
        "--lognorm_alpha", str(float(lognorm_alpha)),
        "--loss_weighting_scheme", loss_weighting_scheme,
        "--prompt_dropout_prob", str(float(prompt_dropout_prob)),
        "--noise_offset", str(float(noise_offset)),
    ]
    
    if use_caption:
        cmd.append("--use_caption")
    if default_caption:
        cmd.extend(["--default_caption", default_caption])
    
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
    
    # 构建状态信息
    if training_mode == "full":
        training_mode_str = "全量训练 (qfloat8)"
    else:
        training_mode_str = "LoKr 微调 (qfloat8)"
    
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
    
    initial_msg = f"""启动 ZImage 训练!

配置信息:
- 模型: {model_id}
- 图片文件夹: {image_folder}
- 训练模式: {training_mode_str}
- 输出目录: {output_dir}
- 训练步数: {num_train_steps}
- 学习率: {learning_rate}
- 分辨率: {resolution}
{timestep_info}
- 使用 Caption: {use_caption}
- Prompt Dropout: {prompt_dropout_prob}
- Noise Offset: {noise_offset}
- 检查点间隔: {checkpoint_every_n_steps}{resume_info}

正在启动训练流程...

"""
    
    for output in common.run_training_process(cmd, initial_msg=initial_msg):
        yield output
    common.training_process = None


def create_page():
    with gr.Column() as page:
        with gr.Row():
            with gr.Column(scale=1):
                training_mode = gr.Radio(
                    label="训练模式",
                    choices=[
                        ("LoKr 微调 (推荐，显存占用低)", "lokr"),
                        ("Full Training 全量训练 (需要更多显存)", "full"),
                    ],
                    value="lokr",
                    info="LoKr 只训练少量参数，Full Training 训练整个模型"
                )
                
                with gr.Row():
                    model_id = gr.Textbox(
                        label="ZImage 模型路径",
                        placeholder="选择 Z-Image 或 Z-Image-Turbo 模型文件夹",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    model_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    model_btn.click(
                        fn=common.select_folder,
                        inputs=model_id,
                        outputs=model_id
                    )
                
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="输出目录",
                        placeholder="选择模型输出保存目录（默认为当前目录下的 output 文件夹）",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    output_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    output_btn.click(
                        fn=common.select_folder,
                        inputs=output_dir,
                        outputs=output_dir
                    )
                
                with gr.Row():
                    image_folder = gr.Textbox(
                        label="训练图片文件夹",
                        placeholder="选择包含训练图片的文件夹（可包含同名 .txt caption 文件）",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    folder_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    folder_btn.click(
                        fn=common.select_folder,
                        inputs=image_folder,
                        outputs=image_folder
                    )
                
                with gr.Row():
                    resume_checkpoint = gr.Textbox(
                        label="从检查点恢复 (可选)",
                        placeholder="选择 checkpoint-xxx 文件夹以继续训练",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    resume_btn = gr.Button("📁", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    resume_btn.click(
                        fn=common.select_folder,
                        inputs=resume_checkpoint,
                        outputs=resume_checkpoint
                    )
                
                use_caption = gr.Checkbox(
                    label="使用 Caption（读取同名 .txt 文件）",
                    value=True,
                )
                
                default_caption = gr.Textbox(
                    label="默认 Caption（无 .txt 文件时使用）",
                    value="",
                    placeholder="可选：为没有 caption 的图片设置默认描述",
                )
                
                with gr.Accordion("高级参数", open=False):
                    num_train_steps = gr.Number(
                        label="训练步数",
                        value=5000,
                        info="总训练步数"
                    )
                    
                    learning_rate = gr.Number(
                        label="学习率",
                        value=1e-4,
                        info="LoKr 建议 1e-4，全量训练建议 1e-5"
                    )
                    
                    resolution = gr.Number(
                        label="训练分辨率",
                        value=1024,
                        info="图片会被缩放到此分辨率（保持宽高比，对齐到32）"
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
                            ("linear - 均匀分布，通用场景", "linear"),
                            ("sigmoid - 集中中间，适合细节和风格", "sigmoid"),
                            ("weighted - 中间权重高，适合蒸馏模型", "weighted"),
                            ("shift - 偏向高噪声，快速学构图", "shift"),
                            ("lognorm_blend - 混合分布，平衡构图和细节", "lognorm_blend"),
                        ],
                        value="sigmoid",
                        info="不同采样方法适合不同训练目标"
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
                        info="偏移程度，越大越偏向高噪声"
                    )
                    
                    lognorm_alpha = gr.Slider(
                        label="LogNorm Alpha (仅 lognorm_blend 有效)",
                        minimum=0.5,
                        maximum=0.9,
                        value=0.75,
                        step=0.05,
                        info="对数正态分布比例，越大越偏向早期时间步"
                    )
                    
                    gr.Markdown("**正则化设置**")
                    
                    loss_weighting_scheme = gr.Dropdown(
                        label="Loss Weighting 方案",
                        choices=[
                            ("none - 无权重，所有时间步等权", "none"),
                            ("sigma_sqrt - 低噪声时权重高，改善细节", "sigma_sqrt"),
                            ("cosmap - 平滑分布，中间时间步权重较高", "cosmap"),
                        ],
                        value="none",
                        info="不同权重方案影响模型对不同噪声级别的学习程度"
                    )
                    
                    prompt_dropout_prob = gr.Slider(
                        label="Prompt Dropout",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        info="训练时随机丢弃 prompt 的概率，用于支持 CFG 推理"
                    )
                    
                    noise_offset = gr.Slider(
                        label="Noise Offset",
                        minimum=0.0,
                        maximum=0.1,
                        value=0.0,
                        step=0.01,
                        info="噪声偏移，可以帮助生成更亮/更暗的图像"
                    )
            
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        start_btn = gr.Button(
                            "开始训练",
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
                        value="等待启动...\n\n配置好参数后点击上方按钮开始训练",
                        interactive=False,
                        lines=28
                    )
                
                start_btn.click(
                    fn=start_training,
                    inputs=[
                        model_id,
                        image_folder,
                        output_dir,
                        resume_checkpoint,
                        training_mode,
                        num_train_steps,
                        checkpoint_every_n_steps,
                        checkpoints_total_limit,
                        learning_rate,
                        resolution,
                        use_caption,
                        default_caption,
                        timestep_type,
                        sigmoid_scale,
                        shift_scale,
                        lognorm_alpha,
                        loss_weighting_scheme,
                        prompt_dropout_prob,
                        noise_offset,
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
