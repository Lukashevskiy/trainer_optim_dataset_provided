import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from instruction_adapter import InstructionPlanDataset
from trainers.grpo_trainer_custom import GRPOTrainer

def main():
    # 1) Пути и названия
    dataset_path = "dataset/optim_super_dataset_EASY_atomic.json"
    model_name   = "Qwen/Qwen2-0.5B-Instruct"

    # 2) Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    # 3) Готовим датасет
    dataset = InstructionPlanDataset(
        path=dataset_path,
        tokenizer=tokenizer,
        max_length=512,
        use_mapping=False,
        min_reward=0.0
    )


    # 5) Конфиг для GRPO
    grpo_config = GRPOConfig(
        output_dir="offline-grpo-run",
        num_iterations=2,
        num_generations=4,
        per_device_train_batch_size=4,
        scale_rewards=True,
    )

    # 6) Инстанцируем OfflineGRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[],                    # не используем external reward_funcs
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_processing_classes=tokenizer,
    )

    # 7) Запускаем обучение
    trainer.train()

if __name__ == "__main__":
    main()
