@echo off
setlocal

set script_dir=%~dp0
set create_venv=true

:parse_args
if "%~1"=="" goto args_parsed
if "%~1"=="--disable-venv" (
    set create_venv=false
)
shift
goto parse_args

:args_parsed

if %create_venv% equ false goto skip_venv
if exist venv goto activate_venv

echo Creating python venv...
python -m venv venv

:activate_venv
call "%script_dir%venv\Scripts\activate"
echo active venv
:skip_venv

echo Installing torch...
echo Select the torch version to install:
echo 1. torch==2.4.1+cu124
echo 2. torch==2.4.1+cu121
echo 3. torch==2.4.1+cu118
set /p torch_choice=Enter your choice (1, 2, or 3): 

if %torch_choice%==1 (
    echo install torch==2.4.1+cu124
    pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
) else if %torch_choice%==2 (
    echo install torch==2.4.1+cu121
    pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
) else if %torch_choice%==3 (
    echo install torch==2.4.1+cu118
    pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Invalid option: %torch_choice%
    exit /b 1
)

echo Installing deps...
pip install -U -r requirements.txt

echo Select dependencies to install:
echo 1. WD Caption (CUDA 12.X)
echo 2. WD Caption (CPU)
echo 3. LLM Caption
echo 4. Huggingface Hub
echo 5. Modelscope Hub

set /p choices=Enter your choices (e.g., 1 2 3): 

for %%i in (%choices%) do (
    if %%i==1 (
        call :install_wd_cu12x
    ) else if %%i==2 (
        call :install_wd_cpu
    ) else if %%i==3 (
        call :install_llm
    ) else if %%i==4 (
        call :install_huggingface
    ) else if %%i==5 (
        call :install_modelscope
    ) else (
        echo Invalid option: %%i
    )
)

goto :eof

:install_wd_cu12x
echo Installing WD Caption (CUDA 12.X) dependencies...
pip install onnxruntime-gpu==1.19.2
goto :continue

:install_wd_cpu
echo Installing WD Caption (CPU) dependencies...
pip install onnxruntime==1.19.2
goto :continue

:install_llm
echo Installing LLM Caption dependencies...
pip install accelerate==0.34.2
pip install bitsandbytes==0.44.0
pip install transformers==4.45.0
pip install sentencepiece==0.2.0
goto :continue

:install_huggingface
echo Installing Huggingface Hub dependencies...
pip install huggingface_hub==0.25.1
goto :continue

:install_modelscope
echo Installing Modelscope Hub dependencies...
pip install modelscope==1.18.1
goto :continue

:continue
goto :eof

:eof
endlocal
echo Installation completed