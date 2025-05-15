import torch
from .grpo_trainer import GRPOTrainer
import math
import torch
from .grpo_trainer import GRPOConfig, RewardFunc
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, Iterable

class OfflineGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[torch.utils.data.Dataset, Iterable[dict]]] = None,
        eval_dataset: Optional[Union[torch.utils.data.Dataset, Iterable[dict], dict[str, Union[torch.utils.data.Dataset, Iterable[dict]]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )

    def _prepare_inputs(self, inputs):
        """
        Если inputs — список примеров из Dataset, сколлируем его в батч-словарь.
        """
        if isinstance(inputs, (list, tuple)):
            batch = {}
            # keys у всех примеров одинаковые
            for key in inputs[0].keys():
                vals = [ex[key] for ex in inputs]
                if isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.stack(vals, dim=0)
                else:
                    batch[key] = vals
            inputs = batch
        return super()._prepare_inputs(inputs)

    def _generate_and_score_completions(self, inputs):
        # 1) берем input и готовые completion из датасета
        prompt_ids      = inputs["input_ids"]
        prompt_mask     = inputs.get("attention_mask", None)
        completion_ids  = inputs["labels"]
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()

        
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        сompletion = completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        old_per_token_logps = self._get_per_token_logps(
            self.model,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
        )

        with torch.no_grad():
            ref_per_token_logps = self._get_per_token_logps(
                self.ref_model,
                prompt_ids,
                prompt_mask,
                completion_ids,
                completion_mask,
            )

        prompts = self.tokenizer.batch_decode(prompt_ids,      skip_special_tokens=True)
        plans   = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        rewards = []
        for instr, plan in zip(prompts, plans):
            entry       = self.mapping.get(instr, {})
            plan_opts   = entry.get("plan_options", [])
            plan_rews   = entry.get("rewards", [])
            r = 0.0
            for po, rv in zip(plan_opts, plan_rews):
                if po.strip() == plan.strip() and rv is not None and not math.isnan(rv):
                    r = rv
                    break
            rewards.append(r)
        rewards = torch.tensor(rewards, device=prompt_ids.device, dtype=torch.float)
        advantages = rewards.unsqueeze(-1).expand_as(old_per_token_logps)

        return {
            "prompt_ids":          prompt_ids,
            "prompt_mask":         prompt_mask,
            "completion_ids":      completion_ids,
            "completion_mask":     completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages":          advantages,
        }