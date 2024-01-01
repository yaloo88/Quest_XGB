@echo off

rem Set the path to your Python executable if it's not in your system PATH
set python_executable=python

rem Get the directory of the batch script (where this script is located)
set script_dir=%~dp0

rem Change the working directory to the project directory
cd /d "%script_dir%"

rem List of Python scripts to run in series
set scripts_to_run=scripts\5_candle_update.py scripts\xgboost_discord.py

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
)

rem Restore the working directory to the original directory
cd ..

rem End of the batch file
exit /b 0
