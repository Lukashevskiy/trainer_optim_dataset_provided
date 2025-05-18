import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, SFTTrainer
from instruction_adapter import InstructionPlanDataset, CachedRefLogitsDataset
from trainers.grpo_trainer_custom import GRPOTrainer
from torch.utils.data import DataLoader
import torch
import pickle
from tqdm import tqdm 
from peft import LoraConfig
import yaml 

def cache_ref_logits(
    dataset_path: str,
    model_name: str,
    cache_path: str,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda"  
):
    # 1) Загрузим датасет и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    dataset   = InstructionPlanDataset(
        path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length,
        use_mapping=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    cache = []
    with torch.no_grad():
        for batch in tqdm(loader):
            # 2) Склеиваем prompt + completion
            prompt_ids      = batch["prompt_ids"].to(device)       # (B, P)
            prompt_mask     = batch["prompt_mask"].to(device)      # (B, P)
            comp_ids        = batch["completion_ids"].to(device)  # (B, C)
            comp_mask       = batch["completion_mask"].to(device) # (B, C)
            rewards         = batch["reward"].tolist()             # list[B]

            input_ids      = torch.cat([prompt_ids,  comp_ids], dim=1)  # (B, P+C)
            attention_mask = torch.cat([prompt_mask, comp_mask], dim=1) # (B, P+C)

            # 3) Forward + log-softmax → log_probs
            outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
            logits    = outputs.logits                               # (B, L, V)
            log_probs = torch.log_softmax(logits, dim=-1)            # (B, L, V)

            # 4) Gather only per-token log-probs for completion part
            #    positions [P .. P+C-1]
            p_len = prompt_ids.size(1)
            c_len = comp_ids.size(1)
            # comp_ids is (B, C), so:
            per_token_logps = log_probs[
                :, 
                p_len : p_len + c_len,      # slice out the completion segment
                :
            ].gather(
                2, comp_ids.unsqueeze(-1)   # (B, C, 1)
            ).squeeze(-1)                   # → (B, C)

            # 5) Добавляем в кеш
            for i in range(input_ids.size(0)):
                cache.append({
                    "prompt_ids":         prompt_ids[i].cpu().tolist(),
                    "attention_mask":     attention_mask[i].cpu().tolist(),
                    "completion_ids":     comp_ids[i].cpu().tolist(),
                    "completion_mask":    comp_mask[i].cpu().tolist(),
                    "ref_per_token_logps": per_token_logps[i].cpu().tolist(),
                    "reward":             rewards[i],
                })

    # 6) Сохраняем в файл
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

def init_peft_config(path):
    lora_config: any
    with open(path, 'r') as f:
        lora_config = yaml.safe_load(f)

    return LoraConfig(
        lora_alpha=lora_config['q_lora']['lora_alpha'],
        lora_dropout=lora_config['q_lora']['lora_dropout'],
        r=lora_config['q_lora']['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
    )

# def init_trainer(self, dataset, model_name):
#     model, tokenizer = self.init_model_and_tokenizer(model_name)
#     training_arguments = self.init_training_args()
#     peft_config = self.init_peft_config()

#     if self.task_type == "sft":
#         return SFTTrainer(
#             model=model,
#             train_dataset=dataset,
#             peft_config=peft_config,
#             dataset_text_field="text",
#             tokenizer=tokenizer,
#             args=training_arguments,
#             packing=self.config['sft']['packing'],
#         )



def main():
    # 1) Пути и названия
    dataset_path = "dataset/optim_super_dataset_EASY_atomic.json"
    dataset_path_pkl = "dataset/optim_super_dataset_EASY_atomic.pkl"
    model_name   = "Qwen/Qwen2-0.5B-Instruct"

    # 2) Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    
    dataset = InstructionPlanDataset(
        path=dataset_path,
        tokenizer=tokenizer,
        max_length=512,
        use_mapping=False,
        min_reward=0.0
    )

    # cache_ref_logits(dataset_path, model_name, "dataset/optim_super_dataset_EASY_atomic.pkl")
    # # # 3) Готовим датасет
    # dataset = CachedRefLogitsDataset(
    #     cache_path=dataset_path_pkl,
    #     pad_token_id=tokenizer.pad_token_id
    # )

    # print(len(dataset))
    # 5) Конфиг для GRPO
    
    peft_config = init_peft_config("peft_config.yaml")
    
    grpo_config = GRPOConfig(
        output_dir="offline-grpo-run",
        num_iterations=2,
        num_generations=2,
        per_device_train_batch_size=4,
        scale_rewards=True,
        max_prompt_length=512,
        max_completion_length=128,
        cache_implementation="sliding_window",
        gradient_accumulation_steps = 1,
    )

    # 6) Инстанцируем OfflineGRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[],                    # не используем external reward_funcs
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_processing_classes=tokenizer,
        peft_config=peft_config
    )

    # 7) Запускаем обучение
    trainer.train()

if __name__ == "__main__":
    main()
