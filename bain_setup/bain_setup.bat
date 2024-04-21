@echo off

call conda env list | findstr /C:"financial_labeling_bain" 1>nul
if errorlevel 1 (
    echo "Creating new environment"
    call conda env create -f bain_environment.yml
    @REM call python -m spacy download en_core_web_sm
) else (
    echo "Activating existing environment"
)

call conda activate financial_labeling_bain

call code .

echo "Done"
