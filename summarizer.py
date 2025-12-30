from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer
MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def summarize_text(text, max_length=150):
    """
    Summarize academic text using T5
    """

    # T5 requires a task prefix
    input_text = "summarize: " + text.strip()

    # Tokenize input
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Generate summary
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode output
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary
