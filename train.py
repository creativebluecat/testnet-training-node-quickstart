import optuna
from dataclasses import dataclass
from demo_test import train_lora  # 请根据你的实际模块名称调整
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
from optuna.visualization import plot_optimization_history, plot_param_importances
import os
from transformers import TrainerCallback
# 设置环境变量
os.environ["HF_TOKEN"] = "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"  # 替换为你的 Hugging Face Token

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float  # 确保是 float 类型以支持小数点

# 定义回调接口  # 引入 TrainerCallback 基类

class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_train_begin(self, args, state, control, **kwargs):
        # 训练开始时调用
        logger.info(f"Training begins for Trial {self.trial.number}")
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        # 每个 epoch 开始时调用
        logger.info(f"Epoch {state.epoch} begins for Trial {self.trial.number}")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # 每个 epoch 结束时调用
        if state.is_world_process_zero:
            eval_loss = state.log_history[-1].get('eval_loss', None)
            if eval_loss is not None:
                self.trial.report(eval_loss, step=state.epoch)
                if self.trial.should_prune():
                    control.should_prune = True
        return control




# Optuna 目标函数（优化目标）
def objective(trial: Trial):
    model_id = 'Qwen/Qwen1.5-1.8B'  # 替换为实际的模型 ID
    context_length = 512  # 假设上下文长度为 512

    # 使用 Optuna 建议的超参数
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 16)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 16)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    lora_rank = trial.suggest_int("lora_rank", 2, 16)
    lora_alpha = trial.suggest_int("lora_alpha", 2, 16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.1, 0.5)

    # 计算有效批量大小
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    max_effective_batch_size = 64  # 假设有效批量大小上限为 64

    if effective_batch_size > max_effective_batch_size:
        # 调整梯度累积步数以保持有效批量大小在上限内
        gradient_accumulation_steps = max(1, max_effective_batch_size // per_device_train_batch_size)
        logger.info(
            f"Adjusted gradient_accumulation_steps to {gradient_accumulation_steps} to keep effective_batch_size <= {max_effective_batch_size}")

    # 定义训练参数对象
    training_args = LoraTrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    # 记录当前试验的超参数
    trial_params = training_args.__dict__
    logger.info(f"Trial {trial.number}: {trial_params}")

    # 创建回调对象
    pruning_callback = OptunaPruningCallback(trial)

    # 训练模型并获取验证损失
    try:
        logger.info("Starting training with the following parameters:")
        eval_loss = train_lora(
            model_id=model_id,
            context_length=context_length,
            training_args=training_args,
            callback=pruning_callback  # 传入回调
        )
        logger.info(f"Training completed. Eval loss: {eval_loss}")
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        raise
    except Exception as e:
        logger.error(f"Error during training in Trial {trial.number}: {e}")
        return float("inf")  # 试验失败时返回一个较大的损失

    # 返回验证损失作为优化目标（Optuna 将最小化这个值）
    return eval_loss


# 运行 Optuna 优化流程
def run_optuna():
    logger.info("Starting Optuna optimization process.")

    # 定义剪枝策略
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)

    # 定义采样策略，设置随机种子以确保可重复性
    sampler = TPESampler(n_startup_trials=5, seed=42)

    # 创建 Optuna 的研究对象，指定存储后端为 SQLite 数据库
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="Lora_Qwen_Model_Optimization",
        storage="sqlite:///optuna_study.db",  # SQLite 数据库用于持久化存储
        load_if_exists=True  # 如果研究对象已存在，则加载它
    )

    # 开始优化，指定试验次数（根据资源情况调整）
    study.optimize(objective, n_trials=50, n_jobs=1, timeout=None)

    # 输出最佳结果
    logger.info("优化完成！")
    logger.info(f"最佳参数: {study.best_params}")
    logger.info(f"最佳验证损失: {study.best_value}")

    # 可视化优化结果
    try:
        # 绘制优化历史
        fig1 = plot_optimization_history(study)
        fig1.savefig("optimization_history.png")
        fig1.show()

        # 绘制参数重要性
        fig2 = plot_param_importances(study)
        fig2.savefig("param_importances.png")
        fig2.show()
    except ImportError:
        logger.warning("Matplotlib 未安装，无法绘制可视化图表。")


if __name__ == "__main__":
    # 运行 Optuna 优化流程
    run_optuna()
