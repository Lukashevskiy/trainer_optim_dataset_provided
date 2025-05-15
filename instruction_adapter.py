import json
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
        max_length: int = 512,
        use_mapping: bool = False,
    ):

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.max_length = max_length
        self.examples = []

        instr_data = raw.get("instructions", {})
        if use_mapping:
            mapping   = raw.get("mapping_plan_to_instruction", {}) # ВОЗМОЖНО ПРИГОДИТЬСЯ)

        for entry in instr_data.values():
            instr   = entry.get("instruction")
            plans   = entry.get("plan_options", [])
            rewards = entry.get("rewards", [])
            
            self.examples.append({
                "instruction": instr,
                "plans": plans,
                "rewards": rewards
            })

        self.examples = [
            ex for ex in self.examples
            if ex["instruction"] and ex["plans"]
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        return {
            "promnt": PROMPT_WITH_FUCNTIONS_HINTS.replace("$INSTRUCTION$", ex['instruction']),
            "completions": ex['plans'],
            "rewards": torch.tensor(ex["rewards"], dtype=torch.float)
        }