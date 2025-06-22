3222222222222222222# %%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# print("device",device)
# We have to check which Torch version for Xformers (2.3 -> 0.0.27)
from torch import __version__; from packaging.version import Version as V
xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
# !pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton
from unsloth import FastLanguageModel
import torch

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")



max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth



'''
Priority Queue:
1. unsloth/Mistral-Nemo-Base-2407-bnb-4bit      (#12B parameters)
2. Phi-3-medium-4k-instruct                     (#14B parameters)
3. gemma-2-27b-bnb-4bit                         (#27B parameters)
'''








model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Mistral-Nemo-Base-2407",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# model.to(device)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# model.to(device)

from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=250):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column_names = ["input_ids", "attention_mask", "text"]  # Manually adding column_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        headline = row["masked headline"]
        number_to_fill = str(row["ans"])  # Convert to string

        input_text = f"Fill in the blank in the headline: '{headline.replace('----', number_to_fill)}' with a number to complete the statement. News: {row['news']}"
        
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),  # Removing batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(),  # Removing batch dimension
            "text": input_text,  # The field used by SFTTrainer
        }




train_data = pd.read_csv("train_num_reas.csv")
# train_data = train_data.head(10)
val_data = pd.read_csv("dev_num_reas.csv")
# val_data = val_data.head(10)

# Create the dataset
train_dataset = NewsDataset(train_data, tokenizer, max_length=350)
val_dataset = NewsDataset(val_data, tokenizer, max_length=350)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


# Setting num_train_epochs to train for a specific number of epochs
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=350,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=10,  # Train for 3 full epochs
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        max_steps=-1,  # No specific max_steps, train based on num_train_epochs
    ),
)

trainer_stats = trainer.train()

# model.save_pretrained("Phi-3-medium-4k-instruct_fintune") # Local saving
# tokenizer.save_pretrained("Phi-3-medium-4k-instruct_fintune")

test_data = pd.read_csv("test_num_reas.csv")
# test_data = test_data.head(50) 

model = FastLanguageModel.for_inference(trainer.model)





from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import pandas as pd

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Define a custom Dataset class for batching
class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=350):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alpaca_prompt = """Instruction: {}\nInput: {}\nOutput: {}"""

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        headline = row["masked headline"]
        input_text = self.alpaca_prompt.format(
            "Fill in the blank in the headline with a number to complete the statement.",
            f"{row['masked headline']} News: {row['news']}",
            ""
        )
        return input_text

# Load the test dataset
test_data = pd.read_csv("test_num_reas.csv")
# test_data = test_data.head(50)  # Adjust based on your dataset size

# Define batch size
batch_size = 8  # Adjust batch size based on your memory constraints

# Create the DataLoader for batching
test_dataset = TestDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Perform inference in batches
all_decoded_predictions = []

for batch in tqdm(test_loader, desc="Processing batches"):
    # Tokenize each batch
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,  # Ensure proper padding for batch processing
        truncation=True,  # Ensure that inputs are truncated to max length
        max_length=350  # Same max_length as used in training
    ).to("cuda")  # Assuming you're using a CUDA-enabled GPU

    # Generate outputs for the batch
    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)

    # Decode the batch of predictions and append them to the list
    decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_decoded_predictions.extend(decoded_predictions)

# Save the predictions into a CSV file
predictions_df = pd.DataFrame({
    'predicted_number': all_decoded_predictions  # Save just the decoded predictions
})
predictions_df.to_csv("Phi-3-medium-4k-instruct.csv", index=False)

print("Decoded predictions saved to 'Phi-3-medium-4k-instruct.csv'")


















# test_dataset = NewsDataset(test_data, tokenizer, max_length=250)

# # Generate predictions on the test dataset
# predictions = trainer.predict(test_dataset=test_dataset)

# # Decode the predictions from token IDs into text
# decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions.predictions.argmax(-1)]

# # Save predictions into a DataFrame
# output_df = pd.DataFrame({
#     'masked_headline': test_data['masked headline'],  # Include the original masked headlines
#     'predicted_number': decoded_predictions,  # The model's predictions
#     'actual_number': test_data['ans']  # The ground truth (actual answer)
# })

# # Save the DataFrame to a CSV file
# output_df.to_csv("prediction.csv", index=False)

# print("Predictions saved to 'prediction.csv'")









# trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         train_dataset=train_dataset,
#         dataset_text_field="text",
#         max_seq_length=250,
#         dataset_num_proc=2,
#         packing=False,
#         args=TrainingArguments(
#              per_device_train_batch_size=2,
#              gradient_accumulation_steps=4,
#              warmup_steps=5,
#              num_train_epochs=3,  # Train for 3 full epochs
#              learning_rate=2e-4,
#              fp16=not is_bfloat16_supported(),
#              bf16=is_bfloat16_supported(),
#              logging_steps=1,
#              optim="adamw_8bit",
#              weight_decay=0.01,
#              lr_scheduler_type="linear",
#              seed=3407,
#              output_dir="outputs",
#              max_steps=-1,  # No specific max_steps, train based on num_train_epochs
#          ),
#      )










