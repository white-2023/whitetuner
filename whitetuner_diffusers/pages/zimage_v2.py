import os
import sys
import json
import gradio as gr

import gui_common as common

PAGE_TITLE = "ZImage-V2 双流架构"


def load_gui_config(checkpoint_path):
    if not checkpoint_path or not checkpoint_path.strip():
        return [gr.update()] * 17
    
    checkpoint_path = checkpoint_path.strip()
    gui_config_path = os.path.join(checkpoint_path, "gui_config.json")
    
    if not os.path.exists(gui_config_path):
        print(f"未找到 GUI 配置文件: {gui_config_path}")
        return [gr.update()] * 17
    
    try:
        with open(gui_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        print(f"已加载 GUI 配置: {gui_config_path}")
        
        return [
            gr.update(value=config.get("model_id", "")),
            gr.update(value=config.get("image_folder", "")),
            gr.update(value=config.get("output_dir", "")),
            gr.update(value=config.get("use_caption", True)),
            gr.update(value=config.get("default_caption", "")),
            gr.update(value=config.get("num_train_steps", 5000)),
            gr.update(value=config.get("learning_rate", 1e-5)),
            gr.update(value=config.get("timestep_type", "sigmoid")),
            gr.update(value=config.get("sigmoid_scale", 1.0)),
            gr.update(value=config.get("shift_scale", 3.0)),
            gr.update(value=config.get("lognorm_alpha", 0.75)),
            gr.update(value=config.get("min_timestep", 0) or 0),
            gr.update(value=config.get("max_timestep", 1000) or 1000),
            gr.update(value=config.get("loss_weighting_scheme", "none")),
            gr.update(value=config.get("prompt_dropout_prob", 0.1)),
            gr.update(value=config.get("n_double_layers", 15)),
            gr.update(value=config.get("n_single_layers", 15)),
        ]
    except Exception as e:
        print(f"加载 GUI 配置失败: {e}")
        return [gr.update()] * 17


def start_training(
    model_id,
    image_folder,
    output_dir,
    use_caption,
    default_caption,
    num_train_steps,
    learning_rate,
    timestep_type,
    sigmoid_scale,
    shift_scale,
    lognorm_alpha,
    min_timestep,
    max_timestep,
    loss_weighting_scheme,
    prompt_dropout_prob,
    noise_offset,
    resolution,
    checkpoint_every_n_steps,
    checkpoints_total_limit,
    n_double_layers,
    n_single_layers,
    resume_from_checkpoint,
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
    
    trainer_script = os.path.join(common.SCRIPT_DIR, "zimage_fixed_v2_trainer.py")
    
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        trainer_script,
        "--model_id", model_id,
        "--image_folder", image_folder,
        "--output_dir", output_dir,
        "--num_train_steps", str(int(num_train_steps)),
        "--learning_rate", str(float(learning_rate)),
        "--timestep_type", timestep_type,
        "--sigmoid_scale", str(float(sigmoid_scale)),
        "--shift_scale", str(float(shift_scale)),
        "--lognorm_alpha", str(float(lognorm_alpha)),
        "--loss_weighting_scheme", loss_weighting_scheme,
        "--prompt_dropout_prob", str(float(prompt_dropout_prob)),
        "--noise_offset", str(float(noise_offset)),
        "--resolution", str(int(resolution)),
        "--checkpoint_every_n_steps", str(int(checkpoint_every_n_steps)),
        "--checkpoints_total_limit", str(int(checkpoints_total_limit)),
        "--n_double_layers", str(int(n_double_layers)),
        "--n_single_layers", str(int(n_single_layers)),
    ]
    
    if min_timestep is not None and min_timestep > 0:
        cmd.extend(["--min_timestep", str(int(min_timestep))])
    if max_timestep is not None and max_timestep < 1000:
        cmd.extend(["--max_timestep", str(int(max_timestep))])
    
    if use_caption:
        cmd.append("--use_caption")
    if default_caption:
        cmd.extend(["--default_caption", default_caption])
    
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint.strip()])
    
    timestep_info = f"- 时间步采样: {timestep_type}"
    if timestep_type == "sigmoid":
        timestep_info += f" (scale={sigmoid_scale})"
    elif timestep_type == "shift":
        timestep_info += f" (scale={shift_scale})"
    elif timestep_type == "lognorm_blend":
        timestep_info += f" (alpha={lognorm_alpha})"
    
    timestep_range_info = ""
    if (min_timestep is not None and min_timestep > 0) or (max_timestep is not None and max_timestep < 1000):
        t_min = int(min_timestep) if min_timestep is not None and min_timestep > 0 else 0
        t_max = int(max_timestep) if max_timestep is not None and max_timestep < 1000 else 1000
        timestep_range_info = f"\n- 时间步范围: [{t_min}, {t_max})"
    
    resume_info = ""
    if resume_from_checkpoint and resume_from_checkpoint.strip():
        resume_info = f"\n- 从检查点恢复: {resume_from_checkpoint}"
    
    initial_msg = f"""使用 accelerate launch 启动 ZImage-V2 双流架构训练!

配置信息:
- 模型: {model_id}
- 图片文件夹: {image_folder}
- 输出目录: {output_dir}
- 训练步数: {num_train_steps}
- 学习率: {learning_rate}
{timestep_info}{timestep_range_info}
- 损失权重: {loss_weighting_scheme}
- Prompt Dropout: {prompt_dropout_prob}
- Noise Offset: {noise_offset}
- 分辨率: {resolution}
- 使用 Caption: {use_caption}{resume_info}

双流架构配置:
- 双流层数: {int(n_double_layers)} (图像和文本独立 FFN)
- 单流层数: {int(n_single_layers)} (图像和文本融合处理)
- 总层数: {int(n_double_layers) + int(n_single_layers)}

架构优势 (相比原版 ZImage):
- 双流设计: 图像和文本在前期有独立的 FFN，只在 Attention 交互
- 独立 Modulation: 图像和文本有独立的调制参数
- 无需重置文本: 文本特征不会漂移

启动命令: accelerate launch zimage_fixed_v2_trainer.py ...

正在启动训练流程...

"""
    
    for output in common.run_training_process(cmd, initial_msg=initial_msg):
        yield output


def create_page():
    with gr.Column() as page:
        gr.Markdown(
            """
            ### ZImage-V2 双流架构训练
            
            **[注意]** 此模式会修改模型架构，约等于重新训练一个全新的 ZImage 模型。训练后的模型与原版 ZImage 不兼容，需要使用配套的推理代码。
            
            基于 FLUX2 的双流设计改进 ZImage，从根本上解决拟合困难问题:
            
            **架构改进:**
            - **双流 Block**: 图像和文本有独立的 FFN，只在 Attention 阶段交互
            - **独立 Modulation**: 图像和文本有独立的调制参数
            - **分阶段处理**: 前 N 层双流处理，后 M 层单流融合
            
            **兼容性:**
            - 可直接加载原版 ZImage/Z-Image-Turbo 权重作为初始化
            - 自动转换: 复制 FFN 权重到 context FFN，复制 modulation 到 txt modulation
            - 保存为新格式，不影响原权重
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    v2_model_id = gr.Textbox(
                        label="ZImage 模型路径",
                        placeholder="选择 Z-Image 或 Z-Image-Turbo 模型文件夹",
                        container=False,
                        max_lines=1,
                        scale=4,
                    )
                    v2_model_btn = gr.Button("...", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    v2_model_btn.click(
                        fn=common.select_folder,
                        inputs=v2_model_id,
                        outputs=v2_model_id
                    )
                
                with gr.Row():
                    v2_output_dir = gr.Textbox(
                        label="输出目录",
                        placeholder="选择模型输出保存目录",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    v2_output_btn = gr.Button("...", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    v2_output_btn.click(
                        fn=common.select_folder,
                        inputs=v2_output_dir,
                        outputs=v2_output_dir
                    )
                
                with gr.Row():
                    v2_image_folder = gr.Textbox(
                        label="训练图片文件夹",
                        placeholder="选择包含训练图片的文件夹（可包含同名 .txt caption 文件）",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    v2_folder_btn = gr.Button("...", scale=0, min_width=40, visible=common.is_local_mode)
                
                if common.is_local_mode:
                    v2_folder_btn.click(
                        fn=common.select_folder,
                        inputs=v2_image_folder,
                        outputs=v2_image_folder
                    )
                
                with gr.Row():
                    v2_resume_checkpoint = gr.Textbox(
                        label="从检查点恢复 (可选)",
                        placeholder="选择 checkpoint-xxx 文件夹以继续训练",
                        container=False,
                        max_lines=1,
                        scale=4
                    )
                    v2_resume_btn = gr.Button("...", scale=0, min_width=40, visible=common.is_local_mode)
                
                v2_use_caption = gr.Checkbox(
                    label="使用 Caption（读取同名 .txt 文件）",
                    value=True,
                )
                
                v2_default_caption = gr.Textbox(
                    label="默认 Caption（无 .txt 文件时使用）",
                    value="",
                    placeholder="可选：为没有 caption 的图片设置默认描述",
                )
                
                with gr.Accordion("双流架构配置", open=True):
                    gr.Markdown("**层数分配** (原版 ZImage 共 30 层)")
                    
                    with gr.Row():
                        v2_n_double_layers = gr.Slider(
                            label="双流层数",
                            minimum=5,
                            maximum=25,
                            value=15,
                            step=1,
                            info="图像和文本独立 FFN 的层数"
                        )
                        v2_n_single_layers = gr.Slider(
                            label="单流层数",
                            minimum=5,
                            maximum=25,
                            value=15,
                            step=1,
                            info="图像和文本融合处理的层数"
                        )
                
                with gr.Accordion("训练参数", open=False):
                    v2_num_train_steps = gr.Number(
                        label="训练步数",
                        value=5000,
                        info="总训练步数"
                    )
                    
                    v2_learning_rate = gr.Number(
                        label="学习率",
                        value=1e-5,
                        info="全量训练建议使用较小的学习率"
                    )
                    
                    gr.Markdown("**时间步采样设置**")
                    
                    v2_timestep_type = gr.Dropdown(
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
                    
                    v2_sigmoid_scale = gr.Slider(
                        label="Sigmoid Scale (仅 sigmoid 有效)",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        info="分布集中程度，越大越集中在中间"
                    )
                    
                    v2_shift_scale = gr.Slider(
                        label="Shift Scale (仅 shift 有效)",
                        minimum=1.0,
                        maximum=6.0,
                        value=3.0,
                        step=0.5,
                        info="偏移程度，越大越偏向高噪声"
                    )
                    
                    v2_lognorm_alpha = gr.Slider(
                        label="LogNorm Alpha (仅 lognorm_blend 有效)",
                        minimum=0.5,
                        maximum=0.9,
                        value=0.75,
                        step=0.05,
                        info="对数正态分布比例，越大越偏向早期时间步"
                    )
                    
                    gr.Markdown("**时间步范围限制**")
                    
                    with gr.Row():
                        v2_min_timestep = gr.Number(
                            label="最小时间步",
                            value=0,
                            minimum=0,
                            maximum=999,
                            info="训练时的最小时间步 (0-999)"
                        )
                        v2_max_timestep = gr.Number(
                            label="最大时间步",
                            value=1000,
                            minimum=1,
                            maximum=1000,
                            info="训练时的最大时间步 (1-1000)"
                        )
                    
                    gr.Markdown("**损失权重**")
                    
                    v2_loss_weighting = gr.Dropdown(
                        label="Loss Weighting 方案",
                        choices=[
                            ("none - 无权重，所有时间步等权", "none"),
                            ("sigma_sqrt - 低噪声时权重高，改善细节", "sigma_sqrt"),
                            ("cosmap - 平滑分布，中间时间步权重较高", "cosmap"),
                        ],
                        value="none",
                        info="不同权重方案影响模型对不同噪声级别的学习程度"
                    )
                    
                    gr.Markdown("**其他参数**")
                    
                    v2_prompt_dropout = gr.Slider(
                        label="Prompt Dropout",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        info="训练时随机丢弃 prompt 的概率，用于支持 CFG 推理"
                    )
                    
                    v2_noise_offset = gr.Slider(
                        label="Noise Offset",
                        minimum=0.0,
                        maximum=0.1,
                        value=0.0,
                        step=0.01,
                        info="噪声偏移，帮助模型更好地生成纯色区域（推荐 0.03-0.05）"
                    )
                    
                    v2_resolution = gr.Number(
                        label="图片分辨率",
                        value=1024,
                        info="训练图片的最大边长"
                    )
                    
                    v2_checkpoint_every = gr.Number(
                        label="检查点保存间隔",
                        value=500,
                        info="每多少步保存一次"
                    )
                    
                    v2_checkpoints_limit = gr.Number(
                        label="检查点保留数量",
                        value=3,
                        info="最多保留多少个"
                    )
            
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        v2_start_btn = gr.Button(
                            "开始 ZImage-V2 训练",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        v2_stop_btn = gr.Button(
                            "停止训练",
                            variant="stop",
                            size="lg",
                            scale=1
                        )
                    
                    v2_status_text = gr.Textbox(
                        label="训练状态",
                        value="等待启动...\n\n配置好参数后点击上方按钮开始 ZImage-V2 训练\n\n双流架构优势:\n- 图像和文本有独立 FFN\n- 独立的 Modulation 参数\n- 无需每层重置文本\n- 更好的拟合能力",
                        interactive=False,
                        lines=28
                    )
                
                config_outputs = [
                    v2_model_id,
                    v2_image_folder,
                    v2_output_dir,
                    v2_use_caption,
                    v2_default_caption,
                    v2_num_train_steps,
                    v2_learning_rate,
                    v2_timestep_type,
                    v2_sigmoid_scale,
                    v2_shift_scale,
                    v2_lognorm_alpha,
                    v2_min_timestep,
                    v2_max_timestep,
                    v2_loss_weighting,
                    v2_prompt_dropout,
                    v2_n_double_layers,
                    v2_n_single_layers,
                ]
                
                v2_start_btn.click(
                    fn=load_gui_config,
                    inputs=v2_resume_checkpoint,
                    outputs=config_outputs
                ).then(
                    fn=start_training,
                    inputs=[
                        v2_model_id,
                        v2_image_folder,
                        v2_output_dir,
                        v2_use_caption,
                        v2_default_caption,
                        v2_num_train_steps,
                        v2_learning_rate,
                        v2_timestep_type,
                        v2_sigmoid_scale,
                        v2_shift_scale,
                        v2_lognorm_alpha,
                        v2_min_timestep,
                        v2_max_timestep,
                        v2_loss_weighting,
                        v2_prompt_dropout,
                        v2_noise_offset,
                        v2_resolution,
                        v2_checkpoint_every,
                        v2_checkpoints_limit,
                        v2_n_double_layers,
                        v2_n_single_layers,
                        v2_resume_checkpoint,
                    ],
                    outputs=v2_status_text,
                    show_progress="full"
                )
                
                v2_stop_btn.click(
                    fn=common.stop_training,
                    inputs=None,
                    outputs=None
                )
                
                if common.is_local_mode:
                    v2_resume_btn.click(
                        fn=common.select_folder,
                        inputs=v2_resume_checkpoint,
                        outputs=v2_resume_checkpoint
                    ).then(
                        fn=load_gui_config,
                        inputs=v2_resume_checkpoint,
                        outputs=config_outputs
                    )
    
    return page

