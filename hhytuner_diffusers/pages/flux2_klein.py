import os
import sys
import gradio as gr

import gui_common as common

PAGE_TITLE = "FLUX.2 Klein Edit 训练"


def start_training(
    model_id,
    target_folder,
    condition_folder_1,
    condition_folder_2,
    condition_folder_3,
    prompt,
    output_dir,
    resume_from_checkpoint,
    num_train_steps,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    timestep_type,
    sigmoid_scale,
    shift_scale,
    lognorm_alpha,
    noise_offset,
    training_mode,
    blocks_to_swap,
    use_pinned_memory,
    learning_rate,
    resolution,
):
    if not model_id:
        yield "[X] 请填写模型路径"
        return
    
    if not target_folder:
        yield "[X] 请填写 Target 文件夹路径"
        return
    
    if not os.path.exists(model_id):
        yield f"[X] 模型路径不存在: {model_id}"
        return
    
    if not os.path.exists(target_folder):
        yield f"[X] Target 文件夹不存在: {target_folder}"
        return
    
    condition_folders_list = []
    for folder in [condition_folder_1, condition_folder_2, condition_folder_3]:
        if folder and folder.strip():
            folder = folder.strip()
            if not os.path.exists(folder):
                yield f"[X] Condition 文件夹不存在: {folder}"
                return
            condition_folders_list.append(folder)
    
    if len(condition_folders_list) == 0:
        yield "[X] 请至少设置一个 Condition 文件夹"
        return
    
    if not output_dir or not output_dir.strip():
        output_dir = os.path.join(common.SCRIPT_DIR, "output")
    
    if not prompt or not prompt.strip():
        prompt = "change image style to anime"
    
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")
    common.start_tensorboard(logdir=tensorboard_logdir, force_restart=True)
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "flux2_klein_trainer.py")
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--model_id", model_id,
        "--target_folder", target_folder,
        "--prompt", prompt,
        "--output_dir", output_dir,
        "--num_train_steps", str(int(num_train_steps)),
        "--checkpoint_every_n_steps", str(int(checkpoint_every_n_steps)),
        "--checkpoints_total_limit", str(int(checkpoints_total_limit)),
        "--timestep_type", timestep_type,
        "--sigmoid_scale", str(float(sigmoid_scale)),
        "--shift_scale", str(float(shift_scale)),
        "--lognorm_alpha", str(float(lognorm_alpha)),
        "--noise_offset", str(float(noise_offset)),
        "--learning_rate", str(float(learning_rate)),
        "--resolution", str(int(resolution)),
    ]
    
    full_training = (training_mode == "full")
    use_block_swap = int(blocks_to_swap) > 0
    
    if full_training:
        cmd.append("--full_training")
    
    if use_block_swap:
        cmd.extend(["--blocks_to_swap", str(int(blocks_to_swap))])
        if use_pinned_memory:
            cmd.append("--use_pinned_memory")
    
    cmd.extend(["--condition_folders"] + condition_folders_list)
    
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
    
    timestep_info = f"- 时间步采样: {timestep_type}"
    if timestep_type == "sigmoid":
        timestep_info += f" (scale={sigmoid_scale})"
    elif timestep_type == "shift":
        timestep_info += f" (scale={shift_scale})"
    elif timestep_type == "lognorm_blend":
        timestep_info += f" (alpha={lognorm_alpha})"
    
    condition_info = f"- Condition 文件夹数量: {len(condition_folders_list)}"
    resume_info = ""
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        resume_info = f"\n- 从检查点恢复: {resume_from_checkpoint}"
    
    if full_training:
        training_mode_str = "全量训练"
    else:
        training_mode_str = "LoKr 微调 (qfloat8 量化)"
    
    block_swap_info = ""
    if use_block_swap:
        training_mode_str += f" + Block Swap ({int(blocks_to_swap)} blocks)"
        block_swap_info = f"\n- Block Swap: {int(blocks_to_swap)} blocks (使用 Adafactor 优化器)"
    
    initial_msg = f"启动 FLUX.2 Klein Edit 训练!\n\n配置信息:\n- Target: {target_folder}\n{condition_info}\n- Prompt: {prompt}\n- 训练模式: {training_mode_str}\n- 输出目录: {output_dir}\n- 训练步数: {num_train_steps}\n- 学习率: {learning_rate}\n- 分辨率: {resolution}\n{timestep_info}\n- Noise Offset: {noise_offset}\n- 检查点间隔: {checkpoint_every_n_steps}{block_swap_info}{resume_info}\n\n正在启动训练流程...\n\n"
    
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
                
                prompt = gr.Textbox(
                    label="提示词",
                    value="change image style to anime",
                    placeholder="change image style to anime",
                )
                with gr.Row():
                    model_id = gr.Textbox(
                        label="模型路径",
                        placeholder="选择 FLUX.2-klein 模型文件夹",
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
                
                with gr.Row():
                    target_folder = gr.Textbox(
                        label="Target 图片文件夹",
                        placeholder="选择训练目标图片文件夹",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    target_label_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    target_label_btn.click(
                        fn=common.select_folder,
                        inputs=target_folder,
                        outputs=target_folder
                    )
                
                with gr.Row():
                    condition_folder_1 = gr.Textbox(
                        label="Condition 文件夹 1",
                        placeholder="选择第一个条件图片文件夹（必填）",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    condition1_label_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    condition1_label_btn.click(
                        fn=common.select_folder,
                        inputs=condition_folder_1,
                        outputs=condition_folder_1
                    )
                
                with gr.Row():
                    condition_folder_2 = gr.Textbox(
                        label="Condition 文件夹 2",
                        placeholder="选择第二个条件图片文件夹（可选）",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    condition2_label_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    condition2_label_btn.click(
                        fn=common.select_folder,
                        inputs=condition_folder_2,
                        outputs=condition_folder_2
                    )
                
                with gr.Row():
                    condition_folder_3 = gr.Textbox(
                        label="Condition 文件夹 3",
                        placeholder="选择第三个条件图片文件夹（可选）",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    condition3_label_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    condition3_label_btn.click(
                        fn=common.select_folder,
                        inputs=condition_folder_3,
                        outputs=condition_folder_3
                    )
                
                with gr.Row():
                    resume_checkpoint = gr.Textbox(
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
                        inputs=resume_checkpoint,
                        outputs=resume_checkpoint
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
                        info="优化器学习率（Block Swap 时使用 Adafactor，否则使用 AdamW8bit）"
                    )
                    
                    resolution = gr.Number(
                        label="训练分辨率",
                        value=1024,
                        info="图片会被缩放到此分辨率（保持宽高比，对齐到16）"
                    )
                    
                    checkpoint_every_n_steps = gr.Number(
                        label="检查点保存间隔",
                        value=500,
                        info="每多少步保存一次"
                    )
                    
                    checkpoints_total_limit = gr.Number(
                        label="检查点保留数量",
                        value=5,
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
                        value="linear",
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
                        maximum=23,
                        value=0,
                        step=1,
                        info="将 single blocks 放在 CPU（最多 23 个），0 表示禁用。启用后自动使用 Adafactor"
                    )
                    
                    use_pinned_memory = gr.Checkbox(
                        label="使用 Pinned Memory",
                        value=True,
                        info="加速 CPU-GPU 数据传输（需要更多内存）"
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
                        target_folder,
                        condition_folder_1,
                        condition_folder_2,
                        condition_folder_3,
                        prompt,
                        output_dir,
                        resume_checkpoint,
                        num_train_steps,
                        checkpoint_every_n_steps,
                        checkpoints_total_limit,
                        timestep_type,
                        sigmoid_scale,
                        shift_scale,
                        lognorm_alpha,
                        noise_offset,
                        training_mode,
                        blocks_to_swap,
                        use_pinned_memory,
                        learning_rate,
                        resolution,
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

