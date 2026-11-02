"""
Task 2: Fine-Tune a Text Classification Model on 10 Custom Tweets Using LoRA
=============================================================================
Assignment Session 3 - Task 2
Trains a pre-trained model on 10 labeled tweets using PEFT LoRA, saves the adapter
weights to disk, and reports the resulting file size.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

# Set seed for reproducibility
torch.manual_seed(42)

# 1. Custom Dataset of 10 sample tweets (5 Positive [label 1], 5 Negative [label 0])
CUSTOM_TWEETS = [
    # Positive Tweets (Label 1)
    {"text": "Absolutely loved the food and lightning-fast delivery! Best biryani in town! ⭐⭐⭐⭐⭐", "label": 1},
    {"text": "The customer support team resolved my refund in less than 2 minutes. Exceptional service!", "label": 1},
    {"text": "Incredible battery life and blazing fast performance on this new phone. Totally worth it!", "label": 1},
    {"text": "Just received my package 2 days earlier than expected. Super impressed with the packing!", "label": 1},
    {"text": "Such a delightful dining experience with authentic Gujarati flavors and warm hospitality.", "label": 1},
    
    # Negative Tweets (Label 0)
    {"text": "Extremely disappointed! Food arrived ice cold, spilled all over the bag, and 1 hour late.", "label": 0},
    {"text": "Worst customer service ever. My order was canceled without reason and no refund issued yet.", "label": 0},
    {"text": "The app keeps crashing whenever I reach the payment checkout screen. Terrible bug!", "label": 0},
    {"text": "Received a defective damaged unit and the seller agent refused to accept the return.", "label": 0},
    {"text": "Horrible travel experience. The flight got delayed by 6 hours without any food or updates.", "label": 0},
]


class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


def main():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = os.path.join(os.path.dirname(__file__), "lora_adapted_tweet_model")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("SESSION 3 - TASK 2: PEFT LoRA FINE-TUNING ON 10 CUSTOM TWEETS")
    print("=" * 80)
    
    # Step 1: Load Tokenizer and Base Model
    print(f"Loading Tokenizer & Base Model: '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Step 2: Configure and Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    print("\nPEFT Model Initialized:")
    model.print_trainable_parameters()
    
    # Step 3: Prepare DataLoader
    dataset = TweetDataset(CUSTOM_TWEETS, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Step 4: Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 3
    model.train()
    
    print(f"\n--- Starting LoRA Training for {epochs} Epochs on 10 Tweets ---")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} | Average Training Loss: {avg_loss:.4f}")

    # Step 5: Save ONLY LoRA Adapter Weights to Disk
    print(f"\n--- Saving LoRA Adapter Weights to: '{output_dir}' ---")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Step 6: Measure and Report File Sizes
    print("\n" + "=" * 80)
    print("SAVED ARTIFACTS AND FILE SIZE REPORT")
    print("=" * 80)
    total_adapter_bytes = 0
    
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            file_bytes = os.path.getsize(file_path)
            total_adapter_bytes += file_bytes
            if file_bytes >= 1024 * 1024:
                size_str = f"{file_bytes / (1024 * 1024):.2f} MB"
            elif file_bytes >= 1024:
                size_str = f"{file_bytes / 1024:.2f} KB"
            else:
                size_str = f"{file_bytes} bytes"
            print(f"  • {file:<30} -> {size_str:>10} ({file_bytes:,} bytes)")
            
    total_adapter_mb = total_adapter_bytes / (1024 * 1024)
    print("-" * 80)
    print(f"Total Adapter Size on Disk: {total_adapter_mb:.2f} MB ({total_adapter_bytes:,} bytes)")
    print("Comparison: A full DistilBERT model weights file is ~268 MB.")
    print(f"Storage Reduction Achieved: ~{(1 - (total_adapter_mb / 268)) * 100:.2f}% savings!")
    print("=" * 80)


if __name__ == "__main__":
    main()

