import pandas as pd
import re
import os

# --- CONFIGURATION ---
input_file = "initial_dataset_multipage.csv"
output_file = "cleaned_dataset.csv"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove Gemini's "Conversational Filler"
    # Removes "Here is the transcription:", "The image contains...", etc.
    text = re.sub(r'^(Here is|This is|Transcription|Answer|The student wrote).*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 2. Remove Markdown bold/italic (*, **, `)
    text = re.sub(r'[*`_]', '', text)

    # 3. Remove Non-ASCII characters (Sinhala letters, weird icons like ➢, ■)
    # We keep standard English letters, numbers, punctuation, and newlines.
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # 4. Fix formatting (remove extra spaces/newlines)
    text = re.sub(r'\n\s*\n', '\n', text)  # Merge multiple blank lines
    text = text.strip()
    
    return text

# --- RUN CLEANER ---
if os.path.exists(input_file):
    print(f"🧹 Cleaning '{input_file}'...")
    df = pd.read_csv(input_file)
    
    # Apply cleaning
    df['clean_text'] = df['extracted_text'].apply(clean_text)
    
    # Swap columns
    df['extracted_text'] = df['clean_text']
    df.drop(columns=['clean_text'], inplace=True)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"✅ DONE! Clean data saved to: '{output_file}'")
    print("   (Use this file for building your memory)")
else:
    print(f"❌ Error: {input_file} not found.")