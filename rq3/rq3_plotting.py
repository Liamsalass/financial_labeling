import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Plotting code')
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to test. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-model_checkpoint_path', type=str, default="rq3_mobilebert_model", help='Specify the relative path to the checkpoint folder.')
    arguments = parser.parse_args()

    # Parse Arguments
    model_name = arguments.model_name
    checkpoint_path = arguments.model_checkpoint_path

    # Verify Arguments
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]
    assert os.path.isdir(checkpoint_path)

    # Loop through Hugging Face model checkpoints to find last one.
    # NOTE: Checkpoint folders are made once per epoch, with the notation checkpoint-<epochs * (num samples/batch size)>
    highest_checkpoint = 0
    highest_checkpoint_folder = ""
    for folder_name in os.listdir(checkpoint_path):
        if folder_name.startswith("checkpoint-"):
            checkpoint_number = int(folder_name.split("-")[1])
            if checkpoint_number > highest_checkpoint:
                highest_checkpoint = checkpoint_number
                highest_checkpoint_folder = folder_name
    
    # Load the train and val information from trainer_state.json in the checkpoint folder
    trainer_state_path = checkpoint_path + "/" + highest_checkpoint_folder + "/trainer_state.json"
    with open(trainer_state_path, "r") as json_file:
        trainer_state_data = json.load(json_file)
    
    # Extracting data from JSON
    num_epochs = (len(trainer_state_data['log_history']) // 2) + 1  # trainer_state reports twice per epoch
    epochs = list(range(1, num_epochs))
    train_loss = [entry['train_loss'] for entry in trainer_state_data['log_history'] if 'train_loss' in entry]
    val_loss = [entry['eval_loss'] for entry in trainer_state_data['log_history'] if 'eval_loss' in entry]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss by Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # TODO: Plot multiple models against each other? Would need to modify command line args and loop through models + store info in a dict, then plot from dict
    # TODO: Plot other metrics by epoch? Like F1, accuracy, etc? Or not necessary
