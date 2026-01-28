

# 导入兼容性符号
from base_trainer import (
    # 工具函数
    fix_windows_encoding,
    get_rank,
    is_main_process,
    print_main,
    apply_quanto_fix,
    pack_latents,
    unpack_latents,
    # 基础类
    BaseTrainerConfig,
    BaseTrainer,
)

from qwen_edit_trainer import (
    # LoKr
    factorization,
    make_kron,
    LokrModule,
    apply_lokr_to_transformer,
    # 配置和数据集
    QwenEditConfig,
    QwenEditDataset,
    # 训练器
    QwenEditTrainer,
    # 主函数
    main,
)

# 旧的类名别名（保持兼容性）
TrainerConfig = QwenEditConfig
QwenEditTrainer = QwenEditTrainer

# 旧的函数名别名
_pack_latents = pack_latents
_unpack_latents = unpack_latents

if __name__ == "__main__":
    main()
