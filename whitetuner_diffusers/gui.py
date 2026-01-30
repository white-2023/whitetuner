import sys
import os
import time
import argparse
import importlib
import pkgutil

import gradio as gr

import gui_common as common
import pages


def discover_pages():
    page_modules = []
    page_titles = []
    
    for _, name, _ in pkgutil.iter_modules(pages.__path__):
        module = importlib.import_module(f'pages.{name}')
        
        create_func = None
        for attr_name in dir(module):
            if attr_name.startswith('create_page'):
                create_func = getattr(module, attr_name)
                break
        
        if create_func and hasattr(module, 'PAGE_TITLE'):
            module.create_page = create_func
            page_modules.append(module)
            page_titles.append(module.PAGE_TITLE)
    
    sorted_pairs = sorted(zip(page_modules, page_titles), key=lambda x: x[0].__name__)
    page_modules, page_titles = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(page_modules), list(page_titles)


def create_gradio_interface():
    with gr.Blocks(title="White Tuner - 图像模型训练器") as demo:
        gr.Markdown(
            """
            # White Tuner - 轻量的图像/视频模型训练器
            
            """
        )
        
        page_modules, page_titles = discover_pages()
        
        with gr.Tabs():
            for module, title in zip(page_modules, page_titles):
                with gr.Tab(title):
                    module.create_page()
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White Tuner - 图像模型训练器 Gradio 界面")
    parser.add_argument("--listen", type=str, default="0.0.0.0", help="服务器监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 服务器端口 (默认: 7860)")
    parser.add_argument("--tensorboard", type=int, default=6006, help="TensorBoard 端口 (默认: 6006)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--share", action="store_true", help="创建公共分享链接")
    parser.add_argument("--server", action="store_true", help="服务器模式：禁用文件夹选择按钮和TensorBoard嵌入")
    parser.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard")
    
    args = parser.parse_args()
    
    if args.server:
        common.is_local_mode = False
        print("[config] 服务器模式（手动指定）：已禁用文件夹选择按钮和 TensorBoard 嵌入")
    else:
        common.is_local_mode = common.detect_gui_environment()
        if common.is_local_mode:
            print("[ok] 检测到图形界面环境，启用本地模式")
        else:
            print("[config] 未检测到图形界面环境，自动切换到服务器模式")
            print("  （文件夹选择按钮和 TensorBoard 嵌入已禁用）")
    
    if args.no_tensorboard:
        common.tensorboard_enabled = False
    
    if common.tensorboard_enabled and common.TENSORBOARD_AVAILABLE:
        print("=" * 80)
        print("启动 TensorBoard...")
        print("=" * 80)
        common.start_tensorboard(port=args.tensorboard)
        
        if common.tensorboard_process is not None:
            print("\n等待 TensorBoard 初始化...")
            max_wait = 10
            for i in range(max_wait):
                time.sleep(1)
                if common.check_port_open(common.tensorboard_port):
                    print(f"[ok] TensorBoard 已就绪")
                    break
                elif i % 3 == 2:
                    print(f"  等待中... ({i+1}/{max_wait}秒)")
            else:
                print(f"[!] TensorBoard 可能需要更长时间启动，请稍后访问")
    elif not common.tensorboard_enabled:
        print("[config] TensorBoard 已禁用")
    elif not common.TENSORBOARD_AVAILABLE:
        print("[!] TensorBoard 未安装，如需使用请运行: pip install tensorboard")
    
    print("\n" + "=" * 80)
    print("启动 Gradio 界面...")
    print(f"访问地址: http://{args.listen}:{args.port}")
    if common.tensorboard_process is not None:
        print(f"TensorBoard: http://localhost:{common.tensorboard_port}")
    print("=" * 80)
    demo = create_gradio_interface()
    
    launch_kwargs = {
        "server_name": args.listen,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
        "inbrowser": not args.no_browser,
    }
    try:
        launch_kwargs["theme"] = gr.themes.Soft()
    except (TypeError, AttributeError):
        pass
    
    try:
        demo.launch(**launch_kwargs)
    finally:
        common.stop_tensorboard()
