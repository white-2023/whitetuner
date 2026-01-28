import os
import sys
import gradio as gr

import gui_common as common

PAGE_TITLE = "GLM-Image"


def start_training(
    model_id,
    task_type,
    image_folder,
    condition_folder,
    target_folder,
    prompt,
    prior_dropout_prob,
    output_dir,
    resume_from_checkpoint,
    num_train_steps,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    learning_rate,
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
):
    if not model_id:
        yield "[X] 请填写模型路径"
        return
    
    if not os.path.exists(model_id):
        yield f"[X] 模型路径不存在: {model_id}"
        return
    
    if task_type == "t2i":
        if not image_folder:
            yield "[X] T2I 模式需要填写图片文件夹路径"
            return
        if not os.path.exists(image_folder):
            yield f"[X] 图片文件夹不存在: {image_folder}"
            return
    else:
        if not condition_folder or not target_folder:
            yield "[X] Edit 模式需要填写条件图像文件夹和目标图像文件夹"
            return
        if not os.path.exists(condition_folder):
            yield f"[X] 条件图像文件夹不存在: {condition_folder}"
            return
        if not os.path.exists(target_folder):
            yield f"[X] 目标图像文件夹不存在: {target_folder}"
            return
    
    if not output_dir or not output_dir.strip():
        output_dir = os.path.join(common.SCRIPT_DIR, "output")
    
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")
    common.start_tensorboard(logdir=tensorboard_logdir, force_restart=True)
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "glm_image_trainer.py")
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--model_id", model_id,
        "--output_dir", output_dir,
        "--task_type", task_type,
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
        "--blocks_to_swap", str(int(blocks_to_swap)),
    ]
    
    if task_type == "t2i":
        cmd.extend(["--image_folder", image_folder])
        if use_caption:
            cmd.append("--use_caption")
            cmd.extend(["--caption_ext", caption_ext])
        if default_caption and default_caption.strip():
            cmd.extend(["--default_caption", default_caption.strip()])
    else:
        cmd.extend(["--condition_folder", condition_folder])
        cmd.extend(["--target_folder", target_folder])
        cmd.extend(["--prior_dropout_prob", str(float(prior_dropout_prob))])
        if prompt and prompt.strip():
            cmd.extend(["--prompt", prompt.strip()])
    
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
    
    block_swap_info = f"\n- Block Swap: {int(blocks_to_swap)} blocks" if blocks_to_swap > 0 else ""
    
    task_str = "T2I 文生图" if task_type == "t2i" else "Edit 图像编辑"
    data_info = f"- 图片文件夹: {image_folder}" if task_type == "t2i" else f"- 条件文件夹: {condition_folder}\n- 目标文件夹: {target_folder}"
    
    initial_msg = f"使用 accelerate launch 启动训练!\n\n配置信息:\n- 模型: GLM-Image\n- 任务类型: {task_str}\n{data_info}\n- 训练模式: Full Training 全量训练\n- 输出目录: {output_dir}\n- 训练步数: {num_train_steps}\n{timestep_info}\n- Noise Offset: {noise_offset}{block_swap_info}\n- 检查点间隔: {checkpoint_every_n_steps}{resume_info}\n\n启动命令: accelerate launch glm_image_trainer.py ...\n\n正在启动训练流程...\n\n"
    
    for output in common.run_training_process(cmd, initial_msg=initial_msg):
        yield output


def create_page():
    with gr.Column() as page:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### GLM-Image 训练")
                gr.Markdown("支持 T2I (文生图) 和 Edit (图像编辑) 两种训练模式。Edit 模式支持多图条件输入：condition/image1.png -> target/image1.png，或 condition/image1_1.png, condition/image1_2.png -> target/image1.png")
                gr.Markdown("**[实验性功能]** Edit 模式会预缓存条件图像的 KV Cache，需要大量硬盘空间（每张图约 2000MB-5000MB）。此功能尚在实验阶段。")
                
                task_type = gr.Radio(
                    label="任务类型",
                    choices=[
                        ("T2I 文生图 - 从文本生成图像", "t2i"),
                        ("Edit 图像编辑 - 从参考图像编辑", "edit"),
                    ],
                    value="t2i",
                    info="选择训练任务类型"
                )
                
                # T2I 模式：默认 Caption 在最上面
                with gr.Column(visible=True) as t2i_caption_group:
                    default_caption = gr.Textbox(
                        label="默认 Caption",
                        value="",
                        placeholder="a photo of a person",
                        info="当图片没有对应的 caption 文件时使用此默认值"
                    )
                
                # Edit 模式：Prompt 在最上面
                with gr.Column(visible=False) as edit_prompt_group:
                    prompt = gr.Textbox(
                        label="编辑 Prompt",
                        value="",
                        placeholder="例如: change image style to anime",
                        info="描述编辑操作的文本提示"
                    )
                
                with gr.Row():
                    model_id = gr.Textbox(
                        label="模型路径",
                        placeholder="选择 GLM-Image 模型文件夹",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    model_label_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    model_label_btn.click(
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
                    output_label_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    output_label_btn.click(
                        fn=common.select_folder,
                        inputs=output_dir,
                        outputs=output_dir
                    )
                
                # T2I 模式的输入
                with gr.Column(visible=True) as t2i_inputs:
                    with gr.Row():
                        image_folder = gr.Textbox(
                            label="图片文件夹",
                            placeholder="选择训练图片文件夹",
                            container=False,
                            max_lines=1,
                            scale=4
                        )
                        image_folder_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                    
                    if common.is_local_mode:
                        image_folder_btn.click(
                            fn=common.select_folder,
                            inputs=image_folder,
                            outputs=image_folder
                        )
                
                # Edit 模式的输入
                with gr.Column(visible=False) as edit_inputs:
                    with gr.Row():
                        condition_folder = gr.Textbox(
                            label="条件图像文件夹",
                            placeholder="选择条件图像文件夹 (参考图像)",
                            container=False,
                            max_lines=1,
                            scale=4
                        )
                        condition_folder_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                    
                    if common.is_local_mode:
                        condition_folder_btn.click(
                            fn=common.select_folder,
                            inputs=condition_folder,
                            outputs=condition_folder
                        )
                    
                    with gr.Row():
                        target_folder = gr.Textbox(
                            label="目标图像文件夹",
                            placeholder="选择目标图像文件夹 (期望输出)",
                            container=False,
                            max_lines=1,
                            scale=4
                        )
                        target_folder_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                    
                    if common.is_local_mode:
                        target_folder_btn.click(
                            fn=common.select_folder,
                            inputs=target_folder,
                            outputs=target_folder
                        )
                    
                    prior_dropout_prob = gr.Slider(
                        label="Prior Dropout 概率",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        info="用于 CFG 训练，表示随机丢弃条件图像的概率"
                    )
                
                # 切换任务类型时显示/隐藏不同的输入
                def update_task_visibility(task):
                    return (
                        gr.update(visible=(task == "t2i")),  # t2i_caption_group
                        gr.update(visible=(task == "edit")),  # edit_prompt_group
                        gr.update(visible=(task == "t2i")),  # t2i_inputs
                        gr.update(visible=(task == "edit")),  # edit_inputs
                    )
                
                task_type.change(
                    fn=update_task_visibility,
                    inputs=[task_type],
                    outputs=[t2i_caption_group, edit_prompt_group, t2i_inputs, edit_inputs]
                )
                
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
                    resume_btn.click(
                        fn=common.select_folder,
                        inputs=resume_from_checkpoint,
                        outputs=resume_from_checkpoint
                    )
                
                with gr.Accordion("高级参数", open=False):
                    use_caption = gr.Checkbox(
                        label="使用 Caption 文件 (T2I 模式)",
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
                        label="学习率",
                        value=1e-4,
                        info="建议 1e-5 到 1e-4"
                    )
                    
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        info="每个 GPU 的批次大小 (Edit 模式固定为 1)"
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
                            ("sigmoid - 集中中间，适合细节和风格", "sigmoid"),
                            ("linear - 均匀分布，通用场景", "linear"),
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
                        value=1.0,
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
                    
                    noise_offset = gr.Slider(
                        label="Noise Offset",
                        minimum=0.0,
                        maximum=0.1,
                        value=0.0,
                        step=0.01,
                        info="噪声偏移，可以帮助生成更亮/更暗的图像"
                    )
                
                with gr.Accordion("显存优化", open=False):
                    blocks_to_swap = gr.Slider(
                        label="Block Swap 数量",
                        minimum=0,
                        maximum=30,
                        value=0,
                        step=1,
                        info="将指定数量的 transformer blocks 交换到 CPU，0 表示禁用。GLM-Image 有 30 个 blocks。"
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
                        value="等待启动...\n\n配置好参数后点击上方按钮开始训练\n\n支持两种训练模式：\n- T2I 文生图：从文本生成图像\n- Edit 图像编辑：从参考图像进行编辑\n\nEdit 模式支持多图条件输入（通过 KV Cache）。",
                        interactive=False,
                        lines=28
                    )
                
                start_btn.click(
                    fn=start_training,
                    inputs=[
                        model_id,
                        task_type,
                        image_folder,
                        condition_folder,
                        target_folder,
                        prompt,
                        prior_dropout_prob,
                        output_dir,
                        resume_from_checkpoint,
                        num_train_steps,
                        checkpoint_every_n_steps,
                        checkpoints_total_limit,
                        learning_rate,
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
