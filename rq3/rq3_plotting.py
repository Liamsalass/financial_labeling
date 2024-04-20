import os
import argparse
import json
import matplotlib.pyplot as plt


def extract_json_data(model_checkpoint_path):
    """
    Function which takes a path to a folder containing HF checkpoints and returns arrays for epochs, train, and val loss
    """
    # Loop through Hugging Face model checkpoints to find last one.
    # NOTE: Checkpoint folders are made once per epoch, with the notation checkpoint-<epochs * (num samples/batch size)>
    highest_checkpoint = 0
    highest_checkpoint_folder = ""
    for folder_name in os.listdir(model_checkpoint_path):
        if folder_name.startswith("checkpoint-"):
            checkpoint_number = int(folder_name.split("-")[1])
            if checkpoint_number > highest_checkpoint:
                highest_checkpoint = checkpoint_number
                highest_checkpoint_folder = folder_name
    
    # Load the train and val information from trainer_state.json in the checkpoint folder
    trainer_state_path = model_checkpoint_path + "/" + highest_checkpoint_folder + "/trainer_state.json"
    with open(trainer_state_path, "r") as json_file:
        trainer_state_data = json.load(json_file)
    
    # Extracting data from JSON
    num_epochs = (len(trainer_state_data['log_history']) // 2) + 1  # trainer_state reports twice per epoch
    epochs_arr = list(range(1, num_epochs))
    train_losses = [entry['train_loss'] for entry in trainer_state_data['log_history'] if 'train_loss' in entry]
    val_losses = [entry['eval_loss'] for entry in trainer_state_data['log_history'] if 'eval_loss' in entry]

    return epochs_arr, train_losses, val_losses


if __name__ == "__main__":
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Plotting Code')
    parser.add_argument('-n_models', type=int, default='1', help='How many models are you plotting? Max: 5')
    arguments = parser.parse_args()
    # TODO: Arguments to plot train, val, or both?

    # Parse Arguments
    n_models = arguments.n_models

    # Verify Arguments
    assert 1 <= n_models <= 5  # Max 5 models plotted at once

    # Read in model names and checkpoint paths
    model_dict = {}
    for idx in range(1, n_models + 1):
        model_name = input("Name of model " + str(idx) + ": ")
        checkpoint_path = input("Relative path to model checkpoint folder: ")
        assert os.path.isdir(checkpoint_path)
        # TODO: Check if the folder has checkpoint-n folders in it? Otherwise prompt for another folder?

        epochs, train_loss, val_loss = extract_json_data(checkpoint_path)
        model_dict[idx] = {
            "Name": model_name,
            "Checkpoint Path": checkpoint_path,
            "Epochs": epochs,
            "Train Loss": train_loss,
            "Validation Loss": val_loss
        }
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for idx, model_data in model_dict.items():
        plt.plot(model_data["Epochs"], model_data["Train Loss"], label=f"{model_data['Name']} Train Loss")
        plt.plot(model_data["Epochs"], model_data["Validation Loss"], label=f"{model_data['Name']} Validation Loss")

    # TODO: Look into plots from papers to see if there should be anything linking the same model on a plot, EG different shapes?

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # TODO: Save plot png
    # TODO: Plot other metrics by epoch? Like F1, accuracy, etc? Or not necessary
    # TODO: Web plot with best results for various models? Possibly have that for test results though?
