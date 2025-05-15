from trl import GRPOConfig
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OfflineGRPOConfig(GRPOConfig):
    """
    Кастомная конфигурация для оффлайн GRPO обучения.
    """
    use_mapping: bool = field(
        default=False,
        metadata={"help": "Использовать mapping_plan_to_instruction для генерации дополнительных примеров"}
    )

    reward_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Отклонить планы с reward ниже этого порога (если задан)"}
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "Максимальная длина токенизированной последовательности"}
    )

    generation_enabled: bool = field(
        default=False,
        metadata={"help": "Использовать ли генерацию модели (для онлайн режима), по умолчанию отключено для оффлайна"}
    )

    def __post_init__(self):
        # можно делать доп. проверки и принты
        if self.generation_enabled:
            print("⚠️ Генерация включена, хотя режим оффлайн — убедись, что trainer настроен правильно.")
