@echo off

call conda env list | findstr /C:"financial_labeling" 1>nul
if errorlevel 1 (
    echo "Creating new environment"
    call conda env create -f environment.yml
) else (
    echo "Activating existing environment"
)

call conda activate financial_labeling

call code .

echo "Done"
