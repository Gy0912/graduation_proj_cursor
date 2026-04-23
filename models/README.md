# models/

**用途**：存放训练产出的适配器/合并权重（论文中写清路径即可复现）。

默认输出（见 `configs/default.yaml`）：

- `outputs/models/lora_only_starcoder2_3b/` — LoRA only（未训练）
- `outputs/models/lora_sft_starcoder2_3b/` — LoRA + SFT
- `outputs/models/lora_dpo_starcoder2_3b/` — LoRA + DPO
- `outputs/models/qlora_only_starcoder2_3b/` — QLoRA only（未训练）
- `outputs/models/qlora_sft_starcoder2_3b/` — QLoRA + SFT
- `outputs/models/qlora_dpo_starcoder2_3b/` — QLoRA + DPO

基座模型权重由 Hugging Face 在线加载，无需手工放入本目录。
