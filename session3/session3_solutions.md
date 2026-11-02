# Session 3: LoRA / PEFT / QLoRA
**Course:** Fine-Tuning & LLM Engineering  
**Workspace:** `e:\TOPS Classes\FINE-TUNING-Assignments\session3`

---

## Task 1: Inspect Trainable Parameters Before and After LoRA (PEFT)

The executable Python script is located at:
[session3/task1_lora_parameters.py](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session3/task1_lora_parameters.py)

### Python Implementation:
```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total > 0 else 0
    return total, trainable, pct

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

total_before, trainable_before, pct_before = count_parameters(base_model)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],
    bias="none"
)
peft_model = get_peft_model(base_model, lora_config)
total_after, trainable_after, pct_after = count_parameters(peft_model)
```

### Verified Terminal Output:
```text
===========================================================================
SESSION 3 - TASK 1: LoRA PARAMETER COMPARISON
===========================================================================
Loading pre-trained base model: 'distilbert-base-uncased-finetuned-sst-2-english' ...

--- [BEFORE APPLYING LoRA] ---
Total Parameters:      66,955,010
Trainable Parameters:  66,955,010
Trainable Percentage:  100.00%

--- [APPLYING LoRA CONFIGURATION] ---
LoRA Rank (r):          8
LoRA Alpha:             16
Target Modules:         {'v_lin', 'q_lin'}
Task Type:              TaskType.SEQ_CLS

--- [AFTER APPLYING LoRA] ---
Total Parameters:      67,694,596
Trainable Parameters:  739,586
Frozen Parameters:     66,955,010
Trainable Percentage:  1.0925%

--- [PEFT BUILT-IN VERIFICATION] ---
trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925
===========================================================================

[KEY INSIGHT] LoRA reduced trainable parameters by a factor of 90.5x!
Only 1.09% of parameters need gradient updates and optimizer state storage.
```

---

## Task 2: Fine-Tuning on 10 Custom Tweets Using LoRA & File Size Report

The executable Python script is located at:
[session3/task2_lora_finetune.py](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session3/task2_lora_finetune.py)  
The saved adapter artifacts are located at:
[session3/lora_adapted_tweet_model](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session3/lora_adapted_tweet_model)

### Custom Labeled Dataset (10 Tweets):
- **Positive Tweets (Label 1):**
  1. *"Absolutely loved the food and lightning-fast delivery! Best biryani in town! ⭐⭐⭐⭐⭐"*
  2. *"The customer support team resolved my refund in less than 2 minutes. Exceptional service!"*
  3. *"Incredible battery life and blazing fast performance on this new phone. Totally worth it!"*
  4. *"Just received my package 2 days earlier than expected. Super impressed with the packing!"*
  5. *"Such a delightful dining experience with authentic Gujarati flavors and warm hospitality."*
- **Negative Tweets (Label 0):**
  1. *"Extremely disappointed! Food arrived ice cold, spilled all over the bag, and 1 hour late."*
  2. *"Worst customer service ever. My order was canceled without reason and no refund issued yet."*
  3. *"The app keeps crashing whenever I reach the payment checkout screen. Terrible bug!"*
  4. *"Received a defective damaged unit and the seller agent refused to accept the return."*
  5. *"Horrible travel experience. The flight got delayed by 6 hours without any food or updates."*

### Training and Adapter Artifact File Size Report:
```text
PEFT Model Initialized:
trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925

--- Starting LoRA Training for 3 Epochs on 10 Tweets ---
Epoch 1/3 | Average Training Loss: 0.0001
Epoch 2/3 | Average Training Loss: 0.0000
Epoch 3/3 | Average Training Loss: 0.0000

--- Saving LoRA Adapter Weights to: 'session3/lora_adapted_tweet_model' ---

================================================================================
SAVED ARTIFACTS AND FILE SIZE REPORT
================================================================================
  • README.md                      ->    5.10 KB (5,218 bytes)
  • adapter_config.json            ->    1.17 KB (1,199 bytes)
  • adapter_model.safetensors      ->    2.82 MB (2,962,192 bytes)
  • tokenizer.json                 ->  694.98 KB (711,659 bytes)
  • tokenizer_config.json          ->  366 bytes (366 bytes)
--------------------------------------------------------------------------------
Total Adapter Size on Disk: 3.51 MB (3,680,634 bytes)
Comparison: A full DistilBERT model weights file is ~268 MB.
Storage Reduction Achieved: ~98.69% savings!
================================================================================
```

---

## Task 3: Benefits of PEFT & Real-World App Use Cases

### Main Benefits of PEFT over Full Model Fine-Tuning:
1. **Dramatic Reduction in GPU Memory (VRAM) & Compute Overhead:** In standard fine-tuning, backpropagation must calculate gradients and store optimizer states (e.g., AdamW's first and second momentum vectors) for 100% of the model’s parameters—requiring 16-20 bytes of memory per parameter. PEFT freezes 99% of the base network and injects low-rank bottleneck matrices ($A$ and $B$), meaning gradients and optimizer states are stored only for ~1% of parameters, allowing consumer GPUs to train large models.
2. **Modular Multi-Tenant Serving with Negligible Storage Footprint:** Instead of storing independent 10GB - 50GB full model checkpoints for every business task, PEFT saves lightweight adapter checkpoints (often only 2 MB to 50 MB). In production, a single frozen foundation model stays permanently resident in GPU VRAM, while hundreds of distinct task adapters are hot-swapped dynamically per user request.
3. **Mitigation of Catastrophic Forgetting:** Full fine-tuning frequently causes neural networks to overwrite their foundational reasoning, math, and syntax knowledge. Because PEFT keeps the original weights intact, the model retains its pre-trained commonsense intelligence while acquiring specialized downstream capabilities.

### Real-World App Examples:
- **App 1 (Instagram): Creator-Specific Writing Style & Persona Adapters**  
  *Why PEFT is ideal:* Instagram hosts millions of creators and brands. Storing a dedicated 7-billion parameter model for every influencer is financially impossible. Using PEFT (LoRA), Instagram can host **one** shared foundation model and maintain separate 5 MB LoRA adapters for each creator, generating captions that mimic their specific vernacular and emoji style on the fly.
- **App 2 (Zomato): City-Specific Regional Culinary Dialect Adapters**  
  *Why PEFT is ideal:* Indian food terminology changes every 100 kilometers (e.g., Kathiyawadi Gujarati in Ahmedabad vs. Malabari cuisine in Kozhikode vs. Mughlai in Lucknow). Zomato can dynamically attach a regional culinary LoRA adapter based on the user's GPS location, ensuring local dish nicknames and dietary rules are understood without deploying separate model infrastructure.

---

## Task 4: QLoRA & RAM Usage Monitoring Comparison

The executable monitoring script is located at:
[session3/task4_qlora_memory_monitor.py](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session3/task4_qlora_memory_monitor.py)  
The generated visual comparison chart is saved at:
[session3/ram_usage_comparison.png](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session3/ram_usage_comparison.png)

### System Resource Measurements & Comparison Matrix:
- **System Total Physical RAM:** 3.90 GB (Windows 64-bit environment)
- **Baseline Idle Process RAM:** 464.81 MB

| Metric / Stage | Standard Full Fine-Tuning | PEFT LoRA (FP32/FP16) | PEFT QLoRA (4-Bit NF4) |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | **66,955,010 (100.0%)** | **739,586 (1.09%)** | **739,586 (1.09%)** |
| **Base Model Precision** | 32-bit Float | 32-bit / 16-bit Float | **4-bit NormalFloat (NF4)** |
| **Peak Process RAM** | **1,090.1 MB** | **728.0 MB** | **349.4 MB** |
| **Optimizer Footprint** | Tracks full 66.9M params (AdamW 2x states) | Tracks only 0.74M adapter params | Tracks only 0.74M adapter params |
| **Memory Savings** | Baseline (0% reduction) | **33.2% RAM reduction** | **67.9% RAM reduction** |

### Task Manager & Resource Monitor Analysis:
During standard full fine-tuning, the Windows Task Manager reveals a sharp memory spike as PyTorch allocates gradient buffers and AdamW momentum tensors for all 66.9 million parameters. In contrast, under LoRA/QLoRA, the gradient graph is truncated strictly at the low-rank linear layers `q_lin` and `v_lin`. When 4-bit quantization (NF4) is applied to the base weights, the model footprint shrinks by over ~67%, enabling models that previously required 24GB GPUs to run effortlessly on low-memory edge devices or modest workstations.

---

## Task 5: ChatGPT/Copilot LoRA Code Snippet, Adaptations & Execution

The executable script is located at:
[session3/task5_lora_hf_snippet.py](file:///e:/TOPS%20Classes/FINE-TUNING-Assignments/session3/task5_lora_hf_snippet.py)

### Raw Code Snippet Generated by ChatGPT:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # Adjusted from generic ['query', 'value']
    bias="none",
    inference_mode=False
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
```

### Necessary Adjustments Made for Our Environment (2-3 Lines):
1. **Target Module Layer Alignment:** ChatGPT originally suggested generic transformer module names (`["query", "value"]` for BERT or `["q_proj", "v_proj"]` for LLaMA), which crashed on DistilBERT with a `ValueError: Target modules not found`. We adjusted the target modules to DistilBERT's exact linear layers: `target_modules=["q_lin", "v_lin"]`.
2. **Explicit TaskType and Head Compatibility:** We explicitly set `task_type=TaskType.SEQ_CLS` in `LoraConfig` and `num_labels=2` in the classification head to ensure the LoRA forward pass correctly routes logits through the sequence classification pooler.
3. **Inference Logit Decoding:** We wrapped the test inputs in `torch.no_grad()` and added softmax decoding to convert raw classification logits into normalized class probabilities and sentiment labels.

### Verified Script Output:
```text
================================================================================
SESSION 3 - TASK 5: CHATGPT/COPILOT LoRA SNIPPET TEST
================================================================================
Loading model: distilbert-base-uncased-finetuned-sst-2-english ...

LoRA Model Successfully Configured!
trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925

Running Inference Verification on Sample Inputs:
  Input: "This product exceeded all my expectations, absolutely fantastic!"
  Logits: [-4.253397464752197, 4.579590797424316] | Predicted: POSITIVE (99.99% confidence)

  Input: "Poor quality, stopped working within two days of delivery."
  Logits: [4.7454047203063965, -3.7515759468078613] | Predicted: NEGATIVE (99.98% confidence)

================================================================================
STATUS: Script executed successfully with 0 errors after target module adjustment!
================================================================================
```

