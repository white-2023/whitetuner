# White Tuner - 轻量图像/视频模型训练器

## 项目简介

White Tuner 是一个基于 Gradio 的图像/视频 AI 模型训练工具，支持多种主流生成模型的微调训练。提供直观的 gradio 界面，支持 LoKr 高效微调和全量训练，内置多种显存优化技术，本项目由老白独立维护（还有claude老师，gpt老师，glm老师，gemini老师，等等....）。

**[特色]** 本训练器是首个同时支持 **多卡训练 + Block Swap** 的开源训练器，可在有限显存下训练大模型。

## 支持的模型

| 模型 | 训练类型 | 说明 |
|------|----------|------|
| FLUX.2 Klein | T2I / Edit | 文生图 / 多图编辑 |
| WAN I2V | I2V | 图生视频 (支持 2.1/2.2) |
| Qwen Image | T2I / Edit | 文生图 / 图像编辑 |
| ZImage | T2I | 文生图 |
| GLM-Image | T2I / Edit | 文生图 / 图像编辑 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/white-2023/whitetuner.git
cd whitetuner
```

### 2. 安装依赖

**Windows 用户需要先手动安装 PyTorch (CUDA 12.8)，或者任意其他torch，例如:**

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

**然后安装其他依赖:**

```bash
pip install -r req.txt
```

主要依赖包括，基本上没有奇怪的库:
- torch, torchvision, torchaudio (PyTorch)
- diffusers, transformers (Hugging Face)
- gradio (Web 界面)
- accelerate (分布式训练)
- bitsandbytes (8bit 优化器)
- optimum-quanto (量化)
- tensorboard (训练监控)

### 3. 启动界面

```bash
cd whitetuner_diffusers
python gui.py
```

启动后访问 `http://localhost:7860` 进入界面

命令行参数:
- `--listen 0.0.0.0` - 监听地址
- `--port 7860` - 界面端口
- `--tensorboard 6006` - TensorBoard 端口
- `--no-browser` - 不自动打开浏览器
- `--server` - 服务器模式 (禁用文件夹选择)


## 项目架构

```
whitetuner/
+-- whitetuner_diffusers/          [核心训练模块]
|   +-- gui.py                   主界面入口
|   +-- gui_common.py            通用工具 (进程管理/TensorBoard)
|   +-- base_trainer.py          基础训练器抽象类
|   +-- lokr.py                  LoKr 微调模块
|   |
|   +-- pages/                   [界面页面]
|   |   +-- flux2_klein.py       FLUX.2 Klein Edit 训练
|   |   +-- flux2_klein_t2i.py   FLUX.2 Klein T2I 训练
|   |   +-- wan.py               WAN I2V 训练
|   |   +-- qwen_image.py        Qwen Image T2I 训练
|   |   +-- qwen_edit.py         Qwen Edit 编辑训练
|   |   +-- zimage.py            ZImage 训练
|   |   +-- glm_image.py         GLM-Image 训练
|   |   +-- tensorboard.py       TensorBoard 监控页面
|   |
|   +-- [训练器脚本]
|   |   +-- flux2_klein_trainer.py
|   |   +-- flux2_klein_t2i_trainer.py
|   |   +-- wan_trainer.py
|   |   +-- qwen_image_trainer.py
|   |   +-- qwen_edit_trainer.py
|   |   +-- zimage_trainer.py
|   |   +-- glm_image_trainer.py
|   |
|   +-- [模型模块]
|       +-- flux2_modules/       FLUX.2 模型实现
|       +-- wan_modules/         WAN 模型实现
|       +-- qwen_modules/        Qwen 模型实现
|       +-- zimage_modules/      ZImage 模型实现
|
+-- inference/                   [推理脚本]
|   +-- flux2_klein_inference.py
|   +-- qwen_image_inference.py
|   +-- zimage_inference.py
|
+-- python/                      [Python 环境]
+-- testdataset/                 [测试数据集]
```


## 界面介绍

### 主界面布局

```
+==============================================================+
|          White Tuner - 轻量的图像/视频模型训练器               |
+==============================================================+
| [FLUX.2 Klein Edit] [FLUX.2 Klein T2I] [WAN I2V] [Qwen T2I]  |
| [ZImage] [GLM-Image] [TensorBoard]                           |
+-------------------------------+------------------------------+
|                               |                              |
|   [参数配置区域]              |   [训练状态区域]             |
|                               |                              |
|   - 训练模式选择              |   +----------------------+   |
|   - 模型路径                  |   | [开始训练] [停止训练]|   |
|   - 输出目录                  |   +----------------------+   |
|   - 数据文件夹                |                              |
|   - Caption 设置              |   训练日志输出...            |
|                               |                              |
|   [高级参数]                  |   - Loss: 0.0234             |
|   - 训练步数                  |   - LR: 1.00e-04             |
|   - 学习率                    |   - Step: 1000/5000          |
|   - 分辨率                    |   - ETA: 15m30s              |
|   - 时间步采样                |                              |
|                               |                              |
|   [显存优化]                  |                              |
|   - Block Swap                |                              |
|   - FP8 量化                  |                              |
|                               |                              |
+-------------------------------+------------------------------+
```

### 数据集准备

所有模型的数据集准备只有两种情况：

#### T2I (文生图) 模式

一个图片文件夹，每张图片配一个同名的 `.txt` 文件作为 caption：

```
image_folder/
    image1.jpg
    image1.txt          [caption 文件，描述图片内容]
    image2.png
    image2.txt
    image3.jpg
    image3.txt
```

#### Edit (图像编辑) 模式

一个 target 文件夹 + 若干个 condition 文件夹，提示词在界面中统一设置（固定提示词）：

```
target_folder/           [目标图像 - 期望的输出结果]
    image1.jpg
    image2.jpg
    image3.jpg

condition_folder_1/      [条件图像1 - 输入参考图]
    image1.jpg           与 target 同名
    image2.jpg
    image3.jpg

condition_folder_2/      [条件图像2 - 可选，多条件输入]
    image1.jpg           支持多图条件: image1_1.jpg, image1_2.jpg
    image2_1.jpg
    image2_2.jpg
```

---

### 训练页面说明

#### WAN I2V LoKr 训练

**[致谢]** 本训练器的大部分代码来自 kohya-ss/musubi-tuner 项目。

训练图生视频模型，支持 WAN 2.1 和 2.2 版本。数据为视频文件或图片序列文件夹。

#### ZImage-V2 双流架构训练

**[注意]** 此模式会修改模型架构，约等于重新训练一个全新的 ZImage 模型。训练后的模型与原版 ZImage 不兼容，需要使用配套的推理代码。

#### GLM-Image Edit 训练

**[实验性功能]** Edit 模式会预缓存条件图像的 KV Cache，需要大量硬盘空间（每张图约 2000MB-5000MB）。此功能尚在实验阶段。

#### TensorBoard 监控

实时监控训练过程: Loss 曲线、学习率变化、权重变化监控


## 训练器架构

```
+=====================================================+
|               BaseTrainer (抽象基类)                |
+=====================================================+
|                                                     |
|  [配置管理]            [训练循环]                   |
|  - BaseTrainerConfig   - train()                    |
|  - 设备/精度设置       - train_step() [抽象]        |
|  - 检查点设置          - 梯度累积                   |
|                        - 多卡同步                   |
|                                                     |
|  [模型管理]            [检查点]                     |
|  - load_models() [抽象] - save_checkpoint() [抽象]  |
|  - Accelerator 准备    - load_checkpoint()          |
|                        - 自动清理旧检查点           |
|                                                     |
|  [优化器]              [日志监控]                   |
|  - AdamW8bit           - TensorBoard                |
|  - Adafactor           - 权重变化监控               |
|  - 学习率调度          - ETA 估算                   |
|                                                     |
+=====================================================+
           |                    |
           v                    v
+-------------------+  +-------------------+
| Flux2KleinTrainer |  | WanTrainer        |
| QwenImageTrainer  |  | ZImageTrainer     |
| GlmImageTrainer   |  | ...               |
+-------------------+  +-------------------+
```

### 时间步采样方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| linear | 均匀分布 | 通用场景 |
| sigmoid | 集中在中间 | 细节、纹理、风格 |
| weighted | 中间权重高 | 蒸馏模型训练 |
| shift | 偏向高噪声 | 快速学习构图 |
| lognorm_blend | 对数正态混合 | 平衡构图和细节 |

```
时间步分布示意 (0=低噪声, 1000=高噪声)

linear:       |====================|  均匀
              0                   1000

sigmoid:      |    =========      |  中间集中
              0                   1000

shift:        |=========          |  偏向高噪声
              0                   1000
```


## LoKr 微调

LoKr (Low-Rank Kronecker) 是一种高效的微调方法:

```
原始权重 W (frozen)
     |
     v
+----+----+
|  W + dW | --> 输出
+----+----+
     ^
     |
dW = kron(W1, W2) * scale

其中 W1, W2 是可训练的小矩阵
```

**优点:**
- 显存占用低 (只训练少量参数)
- 训练速度快
- 输出兼容 LyCORIS 格式


## 显存优化技术

### 1. Block Swap

将部分 Transformer blocks 交换到 CPU:

```
GPU:  [Block 0][Block 1]...[Block N-S]
       ^
       | swap
       v
CPU:  [Block N-S+1]...[Block N]

S = blocks_to_swap 数量
```

### 2. 量化

| 方法 | 说明 | 显存节省 |
|------|------|----------|
| qint8 | INT8 量化 | ~50% |
| qfloat8 / FP8 | FP8 量化 | ~50% |

### 3. 梯度检查点

通过重计算节省中间激活值的显存。


## 检查点格式

```
output/
+-- checkpoints/
|   +-- checkpoint-500/
|   |   +-- transformer/
|   |   |   +-- diffusion_pytorch_model.safetensors
|   |   +-- lokr/                    [LoKr 模式]
|   |   |   +-- lokr.safetensors
|   |   +-- accelerate_state/        [训练状态]
|   |   +-- training_state.json
|   |
|   +-- checkpoint-1000/
|       +-- ...
|
+-- final/
|   +-- transformer/
|   +-- lokr/
|
+-- tensorboard/
    +-- train_5000steps_1gpu_xxx/
```


## 常见问题

### Q: 显存不足怎么办?

1. 使用 LoKr 模式而非 Full Training
2. 启用 Block Swap
3. 降低分辨率
4. 减小 batch size

### Q: 如何选择时间步采样?

- 学风格/纹理: sigmoid
- 学构图: shift
- 通用: linear 或 lognorm_blend

### Q: 训练后如何使用?

LoKr 输出兼容 LyCORIS 格式，可在 ComfyUI 中使用 LoRA Loader 加载。

如果是大模型可以直接复制到comfyui的models/unet 文件夹并加载