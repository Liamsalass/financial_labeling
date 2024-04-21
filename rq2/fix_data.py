import pyarrow as pa
import pandas as pd

# Function to read Arrow stream files and save as text
def arrow_to_text(input_path, output_text_path, output_label_path, append=False):
    # Load the dataset
    with pa.ipc.open_stream(input_path) as reader:
        batches = [batch for batch in reader]
    
    # Combine batches into one table
    table = pa.Table.from_batches(batches)
    df = table.to_pandas()

    # Convert list of tokens to a single space-separated string
    df['text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

    # Write out texts to file
    text_mode = 'a' if append else 'w'
    df['text'].to_csv(output_text_path, index=False, header=False, mode=text_mode)

    # Handle NER tags, converting integers to strings first
    df['labels'] = df['ner_tags'].apply(lambda tags: ' '.join(str(tag) for tag in tags))

    # Write out labels to file
    label_mode = 'a' if append else 'w'
    df['labels'].to_csv(output_label_path, index=False, header=False, mode=label_mode)

# Paths to the Arrow files
train_files = [
    '/home/19jac16/.cache/huggingface/datasets/nlpaueb___finer-139/finer-139/1.0.0/5f5a8eb2a38e8b142bb8ca63f3f9600634cc6c8963e4c982926cf2b48e4e55ff/finer-139-train-00000-of-00002.arrow',
    '/home/19jac16/.cache/huggingface/datasets/nlpaueb___finer-139/finer-139/1.0.0/5f5a8eb2a38e8b142bb8ca63f3f9600634cc6c8963e4c982926cf2b48e4e55ff/finer-139-train-00001-of-00002.arrow'
]

test_file = '/home/19jac16/.cache/huggingface/datasets/nlpaueb___finer-139/finer-139/1.0.0/5f5a8eb2a38e8b142bb8ca63f3f9600634cc6c8963e4c982926cf2b48e4e55ff/finer-139-test.arrow'

# Convert and save train data, append to ensure all training data is in one file
first = True
for file in train_files:
    arrow_to_text(file, 'AttentionXML/data/Finer-139/train_raw_texts.txt', 'AttentionXML/data/Finer-139/train_labels.txt', append=not first)
    first = False

# Convert and save test data
arrow_to_text(test_file, 'AttentionXML/data/Finer-139/test_raw_texts.txt', 'AttentionXML/data/Finer-139/test_labels.txt')
