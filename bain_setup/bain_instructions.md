Instructions on how to load this on Bain Lab computers.
1) Launch VSCode, Python, Git Bash, and Anaconda from AppsAnywhere
2) Clone the repo in the Git Bash terminal
3) Open Anaconda Prompt (Not the powershell version)
`C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\Anaconda Prompt (Anaconda3)`
4) In Anaconda Prompt, navigate to the directory where you cloned the repo, and run bain_setup/bain_setup.bat
5) To run files, use the Anaconda Prompt window (with the financial_labeling environment active), then type `python <filename>.py`, followed by any relevant arguments.
6) While editing, ensure that you select the financial_labelling conda environment when you're in VSCode. (This will also allow you to use cmd directly in VSCode with your conda environment activated. Another way of doing this is through the Anaconda Navigator, where you can open VSCode using the financial_labeling environment.)

Extra Notes:
To update conda environment based on yml file, follow these steps
```
conda activate myenv
conda env update --file local.yml --prune
```

For the bain env (from the bain_setup folder)
```
conda activate financial_labeling_bain
conda env update --file bain_environment.yml --prune
```