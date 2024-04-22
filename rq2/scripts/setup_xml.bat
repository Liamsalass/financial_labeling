@echo off

call conda env list | findstr /C:"xml" 1>nul
if errorlevel 1 (
    echo "Creating new environment"
    call conda env create -f xml.yml
) else (
    echo "Activating existing environment"
)

call conda activate xml

echo "Done"
