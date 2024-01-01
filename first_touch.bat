@echo off

rem Set the path to your Python executable if it's not in your system PATH
set python_executable=python

rem Get the directory of the batch script (where this script is located)
set script_dir=%~dp0

rem Change the working directory to the project directory
cd /d "%script_dir%"

rem Check if 'requirements.txt' exists
if not exist requirements.txt (
    echo 'requirements.txt' not found in the project directory.
    pause
    exit /b 1
)

rem Install required Python packages from 'requirements.txt' if not already installed
%python_executable% -m pip install -r requirements.txt

rem List of Python scripts to run in series
set scripts_to_run=scripts\1_portfolio.py scripts\2_symbol_download.py scripts\3_multi_market_candles.py scripts\4_xgboost.py

rem Loop through the list of scripts and execute them one by one
for %%i in (%scripts_to_run%) do (
    echo Running script: %%i
    %python_executable% "%%i"
    if errorlevel 1 (
        echo Error occurred while running script: %%i
        pause
        exit /b 1
    )
    echo Finished running script: %%i

    rem Procedure message after script execution
    echo -----------------------------------------
    echo Procedure to follow:
    echo 1. Copy the content of market_list\PORTFOLIO.csv to the clipboard.
    echo 2. Paste the content to the console when asked to.
    echo 3. Enter the interval for the candles. 
    echo    - I recommand 'OneMinute,ThreeMinutes,FiveMinutes,FifteenMinutes'.
    echo 4. Open Portfolio_Prediction.pdf and analyse your portfolio's predictions.
    echo -----------------------------------------
)

rem Restore the working directory to the original directory
cd ..

rem End of the batch file
exit /b 0
