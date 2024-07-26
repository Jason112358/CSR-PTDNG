@echo off
setlocal enabledelayedexpansion

call activate PyTorch

REM Set the config dir
set "ENCODER_DIR=./config/write_config/*"
set "CONFIG_DIR=./config/write_config"

REM Set the list of INTERVAL values
set "INTERVALS=second"

REM Set the dir
set "OUTPUT_DIR=./output/multi"

REM Set the path to the global config file
set "GLOBAL_CONFIG_PATH=%CONFIG_DIR%/ENCODER_TYPE/INTERNAL/global_config.yaml"

REM Set the path to the raw config file
set "DATA_CONFIG_PATH=%CONFIG_DIR%/ENCODER_TYPE/INTERNAL/data_config.yaml"

REM Set the path to the train config file
set "TRAIN_CONFIG_PATH=%CONFIG_DIR%/ENCODER_TYPE/INTERNAL/train_config.yaml"

REM Set the path to the draw config file
set "DRAW_CONFIG_PATH=%CONFIG_DIR%/ENCODER_TYPE/INTERNAL/draw_config.yaml"

REM Set the path to the main.py script
set "SCRIPT_PATH=./main.py"

REM Loop through each INTERVAL value
for /D %%E in (%ENCODER_DIR%) do (
    for %%I in (%INTERVALS%) do (
        REM Replace the ENCODER_TYPE value in the config file
        set "NEW_GLOBAL_CONFIG_PATH=!GLOBAL_CONFIG_PATH:ENCODER_TYPE=%%E!"
        set "NEW_DATA_CONFIG_PATH=!DATA_CONFIG_PATH:ENCODER_TYPE=%%E!"
        set "NEW_TRAIN_CONFIG_PATH=!TRAIN_CONFIG_PATH:ENCODER_TYPE=%%E!"
        set "NEW_DRAW_CONFIG_PATH=!DRAW_CONFIG_PATH:ENCODER_TYPE=%%E!"
        REM Replace the INTERVAL value in the config file
        set "NEW_GLOBAL_CONFIG_PATH=!NEW_GLOBAL_CONFIG_PATH:INTERNAL=%%I!"
        set "NEW_DATA_CONFIG_PATH=!NEW_DATA_CONFIG_PATH:INTERNAL=%%I!"
        set "NEW_TRAIN_CONFIG_PATH=!NEW_TRAIN_CONFIG_PATH:INTERNAL=%%I!"
        set "NEW_DRAW_CONFIG_PATH=!NEW_DRAW_CONFIG_PATH:INTERNAL=%%I!"

        echo !NEW_GLOBAL_CONFIG_PATH!
        echo !NEW_DATA_CONFIG_PATH!
        echo !NEW_TRAIN_CONFIG_PATH!
        echo !NEW_DRAW_CONFIG_PATH!
        
        if not exist "%OUTPUT_DIR%/%%E/%%I" (
            mkdir "%OUTPUT_DIR%/%%E/%%I"
            echo %OUTPUT_DIR%/%%E/%%I Created
        ) else (
            echo %OUTPUT_DIR%/%%E/%%I Exists
        )

        REM Execute the main.py script with the updated config file
        mprof run -o %OUTPUT_DIR%/%%E/%%I/%%E-%%I-memory.dat "!SCRIPT_PATH!" --global_config !NEW_GLOBAL_CONFIG_PATH! --data_config !NEW_DATA_CONFIG_PATH! --train_config !NEW_TRAIN_CONFIG_PATH! --draw_config !NEW_DRAW_CONFIG_PATH!
        mprof plot -t %%E-%%I-memory.png -o %OUTPUT_DIR%/%%E/%%I/%%E-%%I-memory.png %OUTPUT_DIR%/%%E/%%I/%%E-%%I-memory.dat

        REM Wait for a few seconds before the next iteration
        timeout /t 5 >nul
    )
)