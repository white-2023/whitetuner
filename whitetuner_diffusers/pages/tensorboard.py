import gradio as gr
import re
import io
import base64
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gui_common as common

PAGE_TITLE = "TensorBoard"


def parse_loss_from_log(log_text: str) -> tuple:
    steps = []
    losses = []
    lrs = []
    
    pattern = r'\|\s*(\d+)/\d+\s*\[.*?loss=([0-9.]+)(?:.*?lr=([0-9.e+-]+))?'
    
    seen_steps = set()
    
    for line in log_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            lr = float(match.group(3)) if match.group(3) else None
            
            if step not in seen_steps:
                seen_steps.add(step)
                steps.append(step)
                losses.append(loss)
                if lr is not None:
                    lrs.append(lr)
    
    return steps, losses, lrs


def parse_weight_monitor_from_log(log_text: str) -> tuple:
    steps = []
    diffs = []
    
    pattern = r'\[Weight Monitor @ step (\d+)\].*?Diff from initial:\s*([0-9.e+-]+)'
    
    matches = re.findall(pattern, log_text, re.DOTALL)
    for match in matches:
        step = int(match[0])
        diff = float(match[1])
        steps.append(step)
        diffs.append(diff)
    
    return steps, diffs


def _percentile_sorted(sorted_values, q):
    if not sorted_values:
        raise ValueError("empty values")
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def calc_ylim_from_percentile(losses, keep_percent):
    if not losses:
        return None, 0
    keep_percent = float(keep_percent)
    if keep_percent <= 0 or keep_percent > 100:
        raise ValueError("keep_percent must be in (0, 100]")
    if keep_percent == 100:
        return None, 0
    trim = (100.0 - keep_percent) / 2.0
    sorted_losses = sorted(losses)
    low = _percentile_sorted(sorted_losses, trim)
    high = _percentile_sorted(sorted_losses, 100.0 - trim)
    padding = (high - low) * 0.05
    outlier_count = sum(1 for v in losses if v < low or v > high)
    return (low - padding, high + padding), outlier_count


def _moving_average_ignore_nan(values, window_size):
    out = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = []
        for v in values[start:i + 1]:
            if not math.isnan(v):
                window.append(v)
        if not window:
            out.append(float("nan"))
        else:
            out.append(sum(window) / len(window))
    return out


def plot_loss_curve(steps, losses, lrs=None, window_size=10, show_lr=False, weight_steps=None, weight_diffs=None, ylim=None, outlier_count=0):
    if not steps or not losses:
        return None
    
    has_weight_data = weight_steps and weight_diffs and len(weight_steps) > 0
    
    if has_weight_data:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 2]})
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
    
    ax1.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.8, label='Loss')
    
    if len(losses) >= window_size:
        smoothed_losses = _moving_average_ignore_nan(losses, window_size)
        ax1.plot(steps, smoothed_losses, 'b-', linewidth=2, label=f'Loss (MA-{window_size})')
    
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', color='b', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    if ylim:
        ax1.set_ylim(ylim[0], ylim[1])
    
    if show_lr and lrs and len(lrs) == len(steps):
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, 'r-', linewidth=1, alpha=0.7, label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='r', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
    
    ax1.legend(loc='upper left')
    title = f'Training Loss (Total {len(steps)} steps)'
    if outlier_count > 0 and ylim:
        title += f' | {outlier_count} 个极端值超出Y轴'
        title += f' | Y轴范围 {ylim[0]:.4f} ~ {ylim[1]:.4f}'
    ax1.set_title(title, fontsize=12)
    
    if has_weight_data:
        ax3.plot(weight_steps, weight_diffs, 'g-o', linewidth=2, markersize=4, label='Weight Diff from Initial')
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Weight Diff', color='g', fontsize=11)
        ax3.tick_params(axis='y', labelcolor='g')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3.set_title('Weight Change Monitor (every 50 steps)', fontsize=12)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" style="width:100%;max-width:1200px;">'


def get_loss_stats(steps, losses, ylim=None, outlier_count=0, weight_steps=None, weight_diffs=None) -> str:
    if not losses:
        return "暂无训练数据"
    
    min_loss = min(losses)
    max_loss = max(losses)
    avg_loss = sum(losses) / len(losses)
    last_n = losses[-10:]
    last_n_avg = sum(last_n) / len(last_n)
    min_step = steps[losses.index(min_loss)]

    stats = f"""
**训练统计:**
- 总步数: {len(steps)}
- Step范围: {steps[0]} ~ {steps[-1]}
- Loss范围: {min_loss:.4f} ~ {max_loss:.4f}
- 平均Loss: {avg_loss:.4f}
- 最终Loss (最后10个点): {last_n_avg:.4f}
- 最低Loss: {min_loss:.4f} @ step {min_step}
"""

    if outlier_count > 0 and ylim:
        stats += f"""
- Y轴限制范围: {ylim[0]:.4f} ~ {ylim[1]:.4f}
- 超出Y轴的极端值: {outlier_count}
"""
    
    if weight_steps and weight_diffs:
        stats += f"""
**权重变化监控:**
- 监控点数: {len(weight_steps)}
- 最新 Weight Diff: {weight_diffs[-1]:.8f} @ step {weight_steps[-1]}
- 最大 Weight Diff: {max(weight_diffs):.8f}
"""
    
    return stats


def create_page():
    with gr.Column() as page:
        def get_tensorboard_info():
            if not common.tensorboard_enabled:
                return """
                ### TensorBoard 实时监控
                
                [!] **TensorBoard 已禁用** (--no-tensorboard)
                
                如需启用 TensorBoard，请去掉 --no-tensorboard 参数重新启动。
                """
            if not common.TENSORBOARD_AVAILABLE:
                return """
                ### TensorBoard 实时监控
                
                [!] **TensorBoard 未安装**
                
                请运行以下命令安装：
                ```
                pip install tensorboard
                ```
                
                安装后重新启动即可使用 TensorBoard。
                """
            if common.is_local_mode:
                return f"""
                ### TensorBoard 实时监控
                
                TensorBoard 运行在端口 **{common.tensorboard_port}**，以下是实时监控界面：
                
                访问地址: http://localhost:{common.tensorboard_port}
                """
            else:
                return f"""
                ### TensorBoard 实时监控
                
                TensorBoard 运行在端口 **{common.tensorboard_port}**
                
                [!] **服务器模式**：iframe 嵌入已禁用，请直接访问 TensorBoard：
                
                - 本机访问: http://localhost:{common.tensorboard_port}
                - 远程访问: http://服务器IP:{common.tensorboard_port}
                
                如需远程访问，请确保服务器防火墙已开放端口 {common.tensorboard_port}
                """
        
        tensorboard_info = gr.Markdown(value=get_tensorboard_info())
        
        tensorboard_available = common.tensorboard_enabled and common.TENSORBOARD_AVAILABLE
        
        if common.is_local_mode and tensorboard_available:
            with gr.Row():
                refresh_btn = gr.Button("刷新 TensorBoard", variant="secondary", size="sm")
            
            def get_tensorboard_iframe():
                return f'<iframe src="http://localhost:{common.tensorboard_port}" width="100%" height="800px" frameborder="0"></iframe>'
            
            tensorboard_frame = gr.HTML(
                value=get_tensorboard_iframe(),
                label="TensorBoard"
            )
            
            def refresh_tensorboard():
                import random
                random_param = random.randint(1, 1000000)
                return f'<iframe src="http://localhost:{common.tensorboard_port}?refresh={random_param}" width="100%" height="800px" frameborder="0"></iframe>'
            
            refresh_btn.click(
                fn=refresh_tensorboard,
                inputs=None,
                outputs=tensorboard_frame
            )
        
        gr.Markdown("---")
        gr.Markdown("### Loss 曲线 (从终端日志提取)")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("**说明**")
                gr.Markdown("从训练终端输出中提取 loss 数据并绘制曲线，支持显示学习率和权重变化监控。")
                gr.Markdown("---")
                gr.Markdown("**图表选项**")
                window_size = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="平滑窗口大小")
                show_lr_checkbox = gr.Checkbox(value=True, label="显示学习率")
                outlier_filter_checkbox = gr.Checkbox(value=True, label="限制Y轴范围(过滤极端值)")
                keep_percent_slider = gr.Slider(minimum=90, maximum=100, value=99, step=1, label="Y轴显示百分位范围")
                gr.Markdown("---")
                loss_stats = gr.Markdown(value="点击「刷新」查看训练数据")
            
            with gr.Column(scale=3):
                with gr.Row():
                    refresh_loss_btn = gr.Button("刷新 Loss 曲线", variant="primary", scale=2)
                    clear_log_btn = gr.Button("清空日志缓存", variant="secondary", scale=1)
                
                log_info = gr.Markdown(value="日志行数: 0")
                loss_plot = gr.HTML(value="<div style='text-align:center;color:#666;padding:50px;border:1px dashed #ccc;border-radius:8px;'>暂无数据，请先开始训练</div>")
        
        def update_loss_plot(window, show_lr, outlier_filter, keep_percent):
            log_text = common.get_loss_log()
            
            line_count = len(log_text.split('\n')) if log_text else 0
            steps, losses, lrs = parse_loss_from_log(log_text)
            weight_steps, weight_diffs = parse_weight_monitor_from_log(log_text)
            
            info = f"日志行数: {line_count} | 有效数据点: {len(steps)}"
            if weight_steps:
                info += f" | 权重监控点: {len(weight_steps)}"
            
            if not steps:
                return (
                    info,
                    "暂无训练数据。请确保训练已开始并产生了 loss 输出。",
                    "<div style='text-align:center;color:#666;padding:50px;'>暂无有效数据</div>"
                )
            
            ylim = None
            outlier_count = 0
            if outlier_filter:
                ylim, outlier_count = calc_ylim_from_percentile(losses, keep_percent)
                if outlier_count > 0 and ylim:
                    info += f" | {outlier_count} 个极端值超出Y轴"

            stats = get_loss_stats(steps, losses, ylim, outlier_count, weight_steps, weight_diffs)
            plot_html = plot_loss_curve(steps, losses, lrs, int(window), show_lr, weight_steps, weight_diffs, ylim, outlier_count)
            
            return info, stats, plot_html
        
        def clear_log():
            common.clear_loss_log()
            return (
                "日志行数: 0 | 缓存已清空",
                "日志缓存已清空",
                "<div style='text-align:center;color:#666;padding:50px;'>日志已清空</div>"
            )
        
        refresh_loss_btn.click(
            fn=update_loss_plot,
            inputs=[window_size, show_lr_checkbox, outlier_filter_checkbox, keep_percent_slider],
            outputs=[log_info, loss_stats, loss_plot]
        )
        
        clear_log_btn.click(
            fn=clear_log,
            inputs=None,
            outputs=[log_info, loss_stats, loss_plot]
        )
    
    return page

