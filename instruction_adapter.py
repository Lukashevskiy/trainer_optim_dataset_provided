import json
import math
import torch
from torch.utils.data import Dataset

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

class InstructionPlanDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
        use_mapping: bool = False,
        min_reward: float = 0.0,
        prompt_template: str = PROMPT_WITH_FUCNTIONS_HINTS,
    ):
        # 1) Читаем JSON и собираем простые примеры
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        instr_data = raw.get('instructions', {})
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for entry in instr_data.values():
            instr = entry.get('instruction')
            plans = entry.get('plan_options', [])
            rewards = entry.get('rewards', [])
            for plan, r in zip(plans, rewards):
                if r is None or math.isnan(r) or r < min_reward:
                    continue
                self.examples.append({
                    'instruction': instr,
                    'completions': plan,
                    'reward': float(r),
                })

        # 2) Один раз токенизируем шаблон без $INSTRUCTION$
        #    он содержит плейсхолдер, который мы заменим динамически
        self.prompt_template = prompt_template
        template_text = prompt_template.replace("$INSTRUCTION$", "")
        toks = tokenizer(
            template_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        self.template_ids  = toks['input_ids'].squeeze(0)      # (P,)
        self.template_mask = toks['attention_mask'].squeeze(0) # (P,)

        # 3) Токенизируем ВСЕ динамические промпты и комплешены разом
        instr_texts = [
            prompt_template.replace("$INSTRUCTION$", ex['instruction'].strip())
            # ex['instruction'].strip()
            for ex in self.examples
        ]
        dyn_tokenized = tokenizer(
            instr_texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
            padding_side="left"
        )
        dyn_ids  = dyn_tokenized['input_ids']      # (N, D)
        dyn_mask = dyn_tokenized['attention_mask'] # (N, D)

        comp_texts = [ex['completions'].strip() for ex in self.examples]
        comp_tokenized = tokenizer(
            comp_texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
            padding_side="left"
        )
        comp_ids  = comp_tokenized['input_ids']      # (N, C)
        comp_mask = comp_tokenized['attention_mask'] # (N, C)

        # 4) Собираем окончательные тензоры в каждом примере
        for i, ex in enumerate(self.examples):
            # склейка шаблона + динамической инструкции
            full_prompt = torch.cat([self.template_ids, dyn_ids[i]], dim=0)
            full_mask   = torch.cat([self.template_mask, dyn_mask[i]], dim=0)
            # потом комплешен (тоже по left-паду)
            full_ids    = torch.cat([full_prompt, comp_ids[i]], dim=0)
            full_mask   = torch.cat([full_mask,   comp_mask[i]], dim=0)

            # теперь обрезаем или пэдим до max_length
            if full_ids.size(0) > max_length:
                # keep last max_length tokens
                full_ids  = full_ids[-max_length:]
                full_mask = full_mask[-max_length:]
            else:
                pad_len = max_length - full_ids.size(0)
                full_ids  = torch.cat(
                    [torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), full_ids],
                    dim=0
                )
                full_mask = torch.cat(
                    [torch.zeros((pad_len,), dtype=torch.long), full_mask],
                    dim=0
                )

            # сохраняем в примере
            ex['prompt_length']     = full_mask.sum().item()  # число ненулевых токенов
            ex['prompt_ids']        = full_ids
            ex['prompt_mask']       = full_mask
            ex['completion_ids']   = comp_ids[i]
            ex['completion_mask']  = comp_mask[i]
            ex['completion']       = comp_texts[i]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # поддерживаем батчи списком/тензором
        if isinstance(idx, (list, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            batch = [self.examples[i] for i in idx]
            out = {}
            for k in batch[0].keys():
                vals = [ex[k] for ex in batch]
                if isinstance(vals[0], torch.Tensor):
                    out[k] = torch.stack(vals, dim=0)
                else:
                    out[k] = vals
            return out

        ex = self.examples[idx]
        return {
            'prompt_length':    ex['prompt_length'],
            'prompt':           ex['instruction'],
            'completion':      ex['completion'],
            'prompt_ids':       ex['prompt_ids'],
            'prompt_mask':      ex['prompt_mask'],
            'completion_ids':  ex['completion_ids'],
            'completion_mask': ex['completion_mask'],
            'reward':           torch.tensor(ex['reward'], dtype=torch.float),
        }


import pickle
import torch
from torch.utils.data import Dataset

class CachedRefLogitsDataset(Dataset):
    def __init__(self, cache_path: str, pad_token_id: int):
        """
        cache_path    — путь к pickle, что вы сохранили в cache_ref_logits.py  
        pad_token_id  — tokenizer.pad_token_id, чтобы можно было пэдить, если нужно  
        """
        with open(cache_path, "rb") as f:
            self.cache = pickle.load(f)
        self.pad = pad_token_id

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        # поддержка батча списка индексов
        if isinstance(idx, (list, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            batch = [self.cache[i] for i in idx]
            out = {}
            for key in batch[0]:
                vals = [ex[key] for ex in batch]
                if isinstance(vals[0], list):
                    # список чисел → Tensor
                    out[key] = torch.tensor(vals, dtype=torch.long if "ids" in key or "mask" in key else torch.float)
                else:
                    # скаляры (reward) → Tensor
                    out[key] = torch.tensor(vals, dtype=torch.float)
            return out

        ex = self.cache[idx]
        return {
            "prompt_ids":           torch.tensor(ex["prompt_ids"],       dtype=torch.long),
            "prompt_mask":      torch.tensor(ex["attention_mask"],   dtype=torch.long),
            "completion_ids":      torch.tensor(ex["completion_ids"],   dtype=torch.long),
            "completion_mask":     torch.tensor(ex["completion_mask"],  dtype=torch.long),
            "ref_per_token_logps": torch.tensor(ex["ref_per_token_logps"], dtype=torch.float),
            "reward":              torch.tensor(ex["reward"],           dtype=torch.float),
        }
