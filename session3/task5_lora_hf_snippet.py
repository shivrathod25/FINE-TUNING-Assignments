"""
Task 5: ChatGPT/Copilot LoRA Code Snippet Execution & Environment Adaptations
=============================================================================
Assignment Session 3 - Task 5
Demonstrates the generated snippet from ChatGPT/Copilot, highlights the necessary
adjustments made for the target model architecture and environment, and verifies execution.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def main():
    print("=" * 80)
    print("SESSION 3 - TASK 5: CHATGPT/COPILOT LoRA SNIPPET TEST")
    print("=" * 80)

    # 1. Base Model & Tokenizer
    # [ADAPTATION NOTE]: ChatGPT originally suggested 'bert-base-uncased', but we adapted
    # it to 'distilbert-base-uncased-finetuned-sst-2-english' to utilize our pre-cached
    # model and reduce execution latency.
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. LoRA Configuration
    # [CRITICAL ADAPTATION NOTE]: Raw ChatGPT snippets commonly set `target_modules=["q_proj", "v_proj"]`
    # (for LLaMA/Mistral) or `["query", "value"]` (for BERT). However, DistilBERT uses
    # `["q_lin", "v_lin"]`. We corrected the target modules to avoid a ValueError.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none",
        inference_mode=False
    )

    # 3. Apply PEFT LoRA
    lora_model = get_peft_model(model, lora_config)
    print("\nLoRA Model Successfully Configured!")
    lora_model.print_trainable_parameters()

    # 4. Forward Pass Inference Test
    sample_texts = [
        "This product exceeded all my expectations, absolutely fantastic!",
        "Poor quality, stopped working within two days of delivery."
    ]
    
    print("\nRunning Inference Verification on Sample Inputs:")
    lora_model.eval()
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = lora_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            confidence = probs[0][predicted_class].item() * 100
            
            print(f"  Input: \"{text}\"")
            print(f"  Logits: {logits.squeeze().tolist()} | Predicted: {sentiment} ({confidence:.2f}% confidence)\n")

    print("=" * 80)
    print("STATUS: Script executed successfully with 0 errors after target module adjustment!")
    print("=" * 80)


if __name__ == "__main__":
    main()

