import os
import sys
import gradio as gr

import gui_common as common

PAGE_TITLE = "Qwen蒸馏"


def start_training(
    model_id,
    text_folder,
    output_dir,
    resume_from_checkpoint,
    num_train_steps,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    learning_rate,
    batch_size,
    min_resolution,
    max_resolution,
    min_teacher_steps,
    max_teacher_steps,
    student_steps,
    distillation_loss_weight,
    training_mode,
    use_fp8,
    blocks_to_swap,
):
    if not model_id:
        yield "[X] 请填写模型路径"
        return
    
    if not text_folder:
        yield "[X] 请填写文本文件夹路径"
        return
    
    if not os.path.exists(model_id):
        yield f"[X] 模型路径不存在: {model_id}"
        return
    
    if not os.path.exists(text_folder):
        yield f"[X] 文本文件夹不存在: {text_folder}"
        return
    
    if not output_dir or not output_dir.strip():
        output_dir = os.path.join(common.SCRIPT_DIR, "output")
    
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")
    common.start_tensorboard(logdir=tensorboard_logdir, force_restart=True)
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "flowmatch_distillation_trainer.py")
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--model_id", model_id,
        "--text_folder", text_folder,
        "--output_dir", output_dir,
        "--num_train_steps", str(int(num_train_steps)),
        "--checkpoint_every_n_steps", str(int(checkpoint_every_n_steps)),
        "--checkpoints_total_limit", str(int(checkpoints_total_limit)),
        "--learning_rate", str(float(learning_rate)),
        "--batch_size", str(int(batch_size)),
        "--min_resolution", str(int(min_resolution)),
        "--max_resolution", str(int(max_resolution)),
        "--min_teacher_steps", str(int(min_teacher_steps)),
        "--max_teacher_steps", str(int(max_teacher_steps)),
        "--student_steps", str(int(student_steps)),
        "--distillation_loss_weight", str(float(distillation_loss_weight)),
        "--blocks_to_swap", str(int(blocks_to_swap)),
        "--training_mode", training_mode,
    ]
    
    full_training = (training_mode == "full")
    
    if use_fp8 and not full_training:
        cmd.append("--use_fp8")
    
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
    
    resume_info = ""
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        resume_info = f"\n- 从检查点恢复: {resume_from_checkpoint}"
    
    training_mode_str = "Full Training (全量训练)" if full_training else "LoKr 微调"
    fp8_info = "\n- FP8: Enabled" if (use_fp8 and not full_training) else ""
    block_swap_info = f"\n- Block Swap: {int(blocks_to_swap)} blocks" if blocks_to_swap > 0 else ""
    
    initial_msg = f"""使用 accelerate launch 启动 FlowMatch Trajectory Distillation 训练!

配置信息:
- 文本文件夹: {text_folder}
- 训练模式: {training_mode_str}
- 输出目录: {output_dir}
- 训练步数: {num_train_steps}

蒸馏配置 (Trajectory 模式 - 纯文本训练):
- 教师步数范围: {min_teacher_steps}-{max_teacher_steps} (从纯噪声推理到最终结果)
- 学生步数: {student_steps} (学习用更少步数达到相同结果)
- 随机分辨率范围: {min_resolution}-{max_resolution}
- 蒸馏损失权重: {distillation_loss_weight}

- 优化器: Adafactor{fp8_info}{block_swap_info}
- 检查点间隔: {checkpoint_every_n_steps}{resume_info}

启动命令: accelerate launch flowmatch_distillation_trainer.py ...

正在启动训练流程...

"""
    
    for output in common.run_training_process(cmd, initial_msg=initial_msg):
        yield output


def create_page():
    with gr.Column() as page:
        gr.Markdown(
            """
            ### FlowMatch Trajectory 蒸馏训练 (纯文本模式)
            
            将教师模型 (30步推理) 的知识蒸馏到学生模型 (4步推理)，实现快速推理。
            
            **纯文本训练**: 只需要提供 .txt 文件，不需要图像。教师从纯噪声开始推理，学生学习用更少步数达到相同结果。
            """
        )
        
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
                        label="模型路径",
                        placeholder="选择 Qwen-Image 或 Qwen-Image-2512 模型文件夹",
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
                    text_folder = gr.Textbox(
                        label="文本文件夹",
                        placeholder="选择包含 .txt 文件的文件夹 (每个文件包含一个 prompt)",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    text_folder_btn = gr.Button("\U0001F4C1", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    text_folder_btn.click(
                        fn=common.select_folder,
                        inputs=text_folder,
                        outputs=text_folder
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
                
                with gr.Accordion("蒸馏参数", open=True):
                    gr.Markdown("**Trajectory 蒸馏**: 教师从纯噪声开始推理N步，学生学习用更少步数达到相同结果")
                    
                    with gr.Row():
                        min_teacher_steps = gr.Number(
                            label="教师最小步数",
                            value=20,
                            info="教师推理步数的最小值"
                        )
                        
                        max_teacher_steps = gr.Number(
                            label="教师最大步数",
                            value=100,
                            info="教师推理步数的最大值"
                        )
                        
                        student_steps = gr.Number(
                            label="学生推理步数",
                            value=4,
                            info="学生模型目标推理步数"
                        )
                    
                    distillation_loss_weight = gr.Slider(
                        label="蒸馏损失权重",
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        info="MSE(学生输出, 教师输出) 的缩放系数"
                    )
                
                with gr.Accordion("基础训练参数", open=False):
                    num_train_steps = gr.Number(
                        label="训练步数",
                        value=5000,
                        info="总训练步数"
                    )
                    
                    learning_rate = gr.Number(
                        label="学习率",
                        value=1e-4,
                        info="优化器学习率 (使用 Adafactor)"
                    )
                    
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        info="每个 GPU 的批次大小"
                    )
                    
                    with gr.Row():
                        min_resolution = gr.Number(
                            label="最小分辨率",
                            value=900,
                            info="随机宽高的最小值"
                        )
                        
                        max_resolution = gr.Number(
                            label="最大分辨率",
                            value=1500,
                            info="随机宽高的最大值"
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
                
                with gr.Accordion("显存优化", open=False):
                    fp8_row = gr.Row()
                    with fp8_row:
                        use_fp8 = gr.Checkbox(
                            label="使用 FP8 量化",
                            value=False,
                            info="使用 FP8 代替 qint8 量化。启用 Block Swap 时自动开启"
                        )
                    
                    blocks_to_swap = gr.Slider(
                        label="Block Swap 数量",
                        minimum=0,
                        maximum=50,
                        value=0,
                        step=1,
                        info="将指定数量的 transformer blocks 交换到 CPU，0 表示禁用"
                    )
                
                def update_mode_visibility(mode):
                    is_lokr = (mode == "lokr")
                    return gr.update(visible=is_lokr)
                
                def update_fp8_on_blockswap(blocks, mode):
                    if mode == "lokr" and blocks > 0:
                        return gr.update(value=True, interactive=False)
                    elif mode == "lokr":
                        return gr.update(interactive=True)
                    return gr.update()
                
                training_mode.change(
                    fn=update_mode_visibility,
                    inputs=[training_mode],
                    outputs=[fp8_row]
                )
                
                blocks_to_swap.change(
                    fn=update_fp8_on_blockswap,
                    inputs=[blocks_to_swap, training_mode],
                    outputs=[use_fp8]
                )
            
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        start_btn = gr.Button(
                            "开始蒸馏训练",
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
                        value="等待启动...\n\n配置好参数后点击上方按钮开始蒸馏训练\n\n纯文本 Trajectory 蒸馏:\n- 不需要图像，只需要 .txt 文件\n- 教师从纯噪声推理30步得到结果\n- 学生学习用4步达到相同结果",
                        interactive=False,
                        lines=28
                    )
                
                start_btn.click(
                    fn=start_training,
                    inputs=[
                        model_id,
                        text_folder,
                        output_dir,
                        resume_from_checkpoint,
                        num_train_steps,
                        checkpoint_every_n_steps,
                        checkpoints_total_limit,
                        learning_rate,
                        batch_size,
                        min_resolution,
                        max_resolution,
                        min_teacher_steps,
                        max_teacher_steps,
                        student_steps,
                        distillation_loss_weight,
                        training_mode,
                        use_fp8,
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
