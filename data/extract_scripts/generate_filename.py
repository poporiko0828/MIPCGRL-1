from typing import Optional


def generate_filename(
    model_name: str,
    finetune: bool,
    instruct: str,
    buffer_ratio: float = 1,
    use_prev: Optional[bool] = None,
) -> str:
    finetune_tag = "_ft" if finetune else ""
    use_prev_tag = f"_prev-{str(use_prev).lower()}" if use_prev else ""
    buffer_ratio_tag = f"_bufratio-{buffer_ratio}" if buffer_ratio < 1 else ""
    return f"embed-{model_name}{finetune_tag}_inst-{instruct}{use_prev_tag}{buffer_ratio_tag}"

