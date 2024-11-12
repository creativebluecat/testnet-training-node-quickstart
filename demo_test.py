import os
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
import logging
from utils.constants import model2template
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
    lora_dropout: int


def train_lora(
        model_id: str, context_length: int, training_args: LoraTrainingArguments, callback=None  # 新增的回调参数
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    logger.info(f"Loading model {model_id}...")

    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
    )

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
    logger.info("Loading training and evaluation datasets...")
    dataset = SFTDataset(
        file="demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # 加载验证数据集
    eval_dataset = SFTDataset(
        file="tmptx0y856x",  # 替换为实际验证数据路径
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,  # 传入验证集
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # 添加回调（如果提供）
    if callback:
        trainer.add_callback(callback)

    # Train model
    logger.info("Starting model training...")
    trainer.train()

    # Evaluate the model on the validation set
    # 评估模型（在回调中已经处理）
    if not callback:
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", float("inf"))
    else:
        eval_loss = float("inf")  # 当使用回调时，eval_loss 由回调处理

    logger.info(f"Training completed with eval_loss: {eval_loss}")

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    logger.info("Training Completed.")
    return eval_loss
