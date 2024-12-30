from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
import torch
import pandas as pd
from data_preparation import load_data

# Load data
train_data_path = 'MATH/train'
test_data_path = 'MATH/test'

print("Loading training data...")
train_df = load_data(train_data_path)
print(f"Training data loaded. Number of samples: {len(train_df)}")

print("Loading testing data...")
test_df = load_data(test_data_path)
print(f"Testing data loaded. Number of samples: {len(test_df)}")

# Prepare the model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Prepare features
def prepare_features(examples):
    questions = [q.strip() for q in examples["problem"]]
    inputs = tokenizer(
        questions,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return inputs

print("Preparing training features...")
train_features = train_df.apply(prepare_features, axis=1)
print("Training features prepared.")

print("Preparing testing features...")
test_features = test_df.apply(prepare_features, axis=1)
print("Testing features prepared.")

# Clear cache
print("Clearing CUDA cache...")
torch.cuda.empty_cache()

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/saved_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced batch size
    per_device_eval_batch_size=8,   # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True  # Enable mixed precision training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_features,
    eval_dataset=test_features,
    tokenizer=tokenizer
)

# Train model
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
trainer.save_model("./models/saved_model")
print("Model saved successfully.")
