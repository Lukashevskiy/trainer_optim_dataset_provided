import json
import pickle
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainers.grpo_trainer import GRPOTrainer, GRPOConfig, RewardFunc
from tqdm import tqdm
from transformers import Trainer



PROMPT_WITH_FUCNTIONS_HINTS = f"""
    You control an agent in a 2D game with siplified Minecraft environment. You will need to provide a detailed step-by-step plan for following the user's instructions. 
    You must include all the preliminary steps that it needs to complete.
    
    You are controlling an agent in a 2D game set within a simplified Minecraft-like environment. 
    The agent starts from scratch with an empty inventory and no gathered resources. 
    Your task is to generate a step-by-step plan that enables the agent to follow a given user instruction.

    What you must do:
    - Break down the instruction into atomic actions the agent needs to perform.
    - Include all necessary preliminary steps, such as gathering or crafting resources.
    - Assume the agent has nothing at the beginning — you must plan from the ground up.
    - Output your answer as a Python list of strings.
    - Each string must represent one atomic skill invocation, written on a separate line.

    Format for each step:
    "skill_name(arg1 = value1, arg2 = value2, ...)"
    - skill_name: the name of the primitive skill or action the agent will execute.
    - Inside the parentheses, list all required arguments with their names and corresponding values.

    Example:
    gather_resource(resource_type = wood)
    
    Each of the step agents will be implemented without knowledge of what it did before, so it can only rely on observation and the current step. Therefore, each step must be self-sufficient and not require knowledge of past steps.

    Existed skills:
    {{'explore': "'object'"}}
    {{'gather_resource': "'resource_type'"}}, 
    {{'place_item': "'item_type'}},
    {{'create_item': "'item_type_to_craft'"}},
    {{'defeat_enemy': "'enemy_type'"}},
    {{'eat': "'food_type'"}},


    Existed arguments:
    object: world
    resource_type: wood, stone, coal, diamond, iron, plant, water
    item_type: stone, table, plant, furnace
    item_type_to_craft: table, wooden sword, wooden_pickaxe, stone_sword, stone_pickaxe, iron_sword, iron_pickaxe
    enemy_type: zombie, skeleton, cow
    food_type: cow, plant, water
    
    "If the instruction doesn't specify what the agent needs to do and is more general—like 'Explore the world' or 'Go out and examine the world around you'—send explore(object=world). In this case, the plan should consist of only one step: "explore(object=world)"."

    Send your answer as a python list.
    Instruction: Make a pickaxe from wood
    Answer: 
    ["gather_resource(resource_type = wood)",
    "gather_resource(resource_type = wood)",
    "create_item(item_type = table)", 
    "gather_resource(resource_type = wood", 
    "gather_resource(resource_type = wood", 
    "create_item(item_type = wooden_pickaxe)"]

    Send your answer as a python list.
    Instruction: $INSTRUCTION$  
    Answer: 
    """



# 1. Оригинальный Dataset для инструкций и планов
class InstructionPlanDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512, use_mapping=False, min_reward=0.0):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        instr_data = raw.get('instructions', {})
        # mapping   = raw.get('mapping_plan_to_instruction', {})
        for entry in instr_data.values():
            instr = entry.get('instruction')
            plans = entry.get('plan_options', [])
            rewards = entry.get('rewards', [])
            for plan, r in zip(plans, rewards):
                if r is None or math.isnan(r) or r<min_reward: continue
                self.examples.append({'promt_instr': instr, 'complections': plan, 'reward': float(r)})
        self.examples = [ex for ex in self.examples if ex['promt_instr'] and ex['complections']]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = PROMPT_WITH_FUCNTIONS_HINTS.replace("$INSTRUCTION$", ex['promt_instr'].strip())
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        attn_mask = tokenized['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': input_ids.clone(),  # teacher-forcing
            'reward': torch.tensor(ex['reward'], dtype=torch.float)
        }

# 2. Предпроцессинг: кеширование logits
def cache_logits(json_path, cache_path, model_name, batch_size=6, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).eval()
    dataset   = InstructionPlanDataset(json_path, tokenizer, max_length=max_length)
    loader    = DataLoader(dataset, batch_size=batch_size)
    cache = []
    with torch.no_grad():
        for batch in tqdm(loader):
            in_ids = batch['input_ids']
            attn   = batch['attention_mask']
            labels = batch['labels']
            rewards= batch['reward']
            outputs= model(input_ids=in_ids, attention_mask=attn)
            logits = outputs.logits
            logp   = torch.log_softmax(logits, dim=-1)
            per_tok= logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            ref_tok= per_tok.clone()  # ref=model
            for i in range(in_ids.size(0)):
                cache.append({
                    'input_ids': in_ids[i].tolist(),
                    'attention_mask': attn[i].tolist(),
                    'labels': labels[i].tolist(),
                    'old_per_token_logps': per_tok[i].tolist(),
                    'ref_per_token_logps': ref_tok[i].tolist(),
                    'reward': rewards[i].item()
                })
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cached {len(cache)} examples to {cache_path}")

# 3. Dataset для закешированных logits
class CachedLogitsDataset(Dataset):
    def __init__(self, cache_path, pad_token_id):
        with open(cache_path, 'rb') as f:
            self.cache = pickle.load(f)
        self.pad = pad_token_id
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        ex = self.cache[idx]
        return {
            'input_ids': torch.tensor(ex['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(ex['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(ex['labels'], dtype=torch.long),
            'old_per_token_logps': torch.tensor(ex['old_per_token_logps'], dtype=torch.float),
            'ref_per_token_logps': torch.tensor(ex['ref_per_token_logps'], dtype=torch.float),
            'reward': torch.tensor(ex['reward'], dtype=torch.float)
        }

# 4. Offline GRPOTrainer override
class OfflineGRPOTrainer(GRPOTrainer):
    def _preprepare_inputs(self, inputs):
        # Коллатим список примеров в словарь батча
        if isinstance(inputs, (list, tuple)):
            batch = {}
            for key in inputs[0].keys():
                vals = [ex[key] for ex in inputs]
                if isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.stack(vals, dim=0)
                else:
                    batch[key] = vals
            inputs = batch
        return inputs
    
    def _generate_and_score_completions(self, inputs):
        batched_inputs = self._preprepare_inputs(inputs)
        prompt_ids = batched_inputs['input_ids']
        prompt_mask= batched_inputs.get('attention_mask', None)
        completion_ids = batched_inputs['labels']
        completion_mask= (completion_ids != self.tokenizer.pad_token_id).long()
        old_per = batched_inputs['old_per_token_logps']
        ref_per = batched_inputs['ref_per_token_logps']
        
        raw_rewards = batched_inputs["reward"].view(-1)            # (batch,)

        # 1) нормируем по среднему и стандартному отклонению
        adv_mean = raw_rewards.mean()
        adv_std  = raw_rewards.std(unbiased=False)
        advantages_norm = (raw_rewards - adv_mean) / (adv_std + 1e-8)

        # 2) растягиваем на все токены completion
        # old_per_token_logps имеет форму (batch, seq_len)
        advantages = advantages_norm.unsqueeze(-1).expand_as(old_per)
        return {
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
            'old_per_token_logps': old_per,
            'ref_per_token_logps': ref_per,
            'advantages': advantages
        }

# 5. Главная функция для обучения
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default="dataset/optim_super_dataset_EASY_atomic.json")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-0.5B-Instruct')
    parser.add_argument('--cache', type=str, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--update_cache_logits', type=bool, default=False)

    args = parser.parse_args()
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    
    if args.update_cache_logits:
        cache_logits(args.json, args.cache, args.model, batch_size=args.batch)

    # Загружаем кеш и готовим trainer
    cached_refs_logits_dataset   = CachedLogitsDataset(args.cache, tokenizer.pad_token_id)
    config    = GRPOConfig()
    trainer   = OfflineGRPOTrainer(
        model=args.model,
        reward_funcs=[],
        args=config,
        train_dataset=cached_refs_logits_dataset,
    )
    trainer.train()
