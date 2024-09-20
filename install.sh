#!/usr/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
create_venv=true

while [ -n "$1" ]; do
    case "$1" in
        --disable-venv)
            create_venv=false
            ;;
    esac
    shift
done

activate_venv() {
    if [[ -n "$COMSPEC" ]]; then
        source "$script_dir/venv/Scripts/activate"
    else
        source "$script_dir/venv/bin/activate"
    fi
    echo "active venv"
}

if [ -d "$script_dir/venv" ]; then
    activate_venv
else
    if $create_venv; then
        if [[ -n "$COMSPEC" ]]; then
            python -m venv "$script_dir/venv"
        else
            python3 -m venv "$script_dir/venv"
        fi
        activate_venv
    else
        echo "Skipping venv creation and activation"
    fi
fi

echo "Installing torch..."

cuda_version=$(nvidia-smi | grep -oiP 'CUDA Version: \K[\d\.]+')

if [ -z "$cuda_version" ]; then
    cuda_version=$(nvcc --version | grep -oiP 'release \K[\d\.]+')
fi
cuda_major_version=$(echo "$cuda_version" | awk -F'.' '{print $1}')
cuda_minor_version=$(echo "$cuda_version" | awk -F'.' '{print $2}')

echo "CUDA Version: $cuda_version"


if (( cuda_major_version > 12 || (cuda_major_version == 12 && cuda_minor_version >= 4) )); then
    echo "install torch==2.4.1+cu124"
    pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
    elif (( cuda_major_version == 12 && cuda_minor_version >= 1 )); then
    echo "install torch==2.4.1+cu121"
    pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
elif (( cuda_major_version == 11 && cuda_minor_version >= 8 )); then
    echo "install torch==2.4.1+cu118"
    pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
elif (( cuda_major_version == 11 && cuda_minor_version >= 6 )); then
    echo "install torch 1.12.1+cu116"
    pip install torch==1.12.1+cu116 --index-url https://download.pytorch.org/whl/cu116
elif (( cuda_major_version == 11 && cuda_minor_version >= 2 )); then
    echo "install torch 1.12.1+cu113"
    pip install torch==1.12.1+cu113 --index-url https://download.pytorch.org/whl/cu116
else
    echo "Unsupported cuda version:$cuda_version"
    exit 1
fi

echo "Installing deps..."
pip install -U -r requirements.txt
echo "Select dependencies to install:"
echo "1. WD Caption"
echo "2. Joy Caption"
echo "3. Huggingface Hub"
echo "4. Modelscope Hub"

read -p "Enter your choices (e.g., 1 2 3): " choices

install_wd() {
    echo "Installing WD Caption dependencies..."
    pip install onnx==1.16.2
}

install_joy() {
    echo "Installing Joy Caption dependencies..."
    pip install accelerate==0.33.0
    pip install bitsandbytes==0.43.3
    pip install transformers==4.44.2
    pip install sentencepiece==0.2.0
}

install_huggingface() {
    echo "Installing Huggingface Hub dependencies..."
    pip install huggingface_hub==0.24.6
}

install_modelscope() {
    echo "Installing Modelscope Hub dependencies..."
    pip install modelscope==1.17.1
}

install_wd_cu12x() {
    echo "Installing WD Caption (CUDA 12.X) dependencies..."
    pip install onnxruntime-gpu==1.19.0
}

install_wd_cpu() {
    echo "Installing WD Caption (CUDA 12.X) dependencies..."
    pip install onnxruntime==1.19.0
}

for choice in $choices; do
    case $choice in
        1)
            install_wd
            echo "Select CUDA version for WD Caption:"
            echo "1. CUDA 12.X"
            echo "2. CPU"
            read -p "Enter your choice (1, 2): " cuda_choice

            case $cuda_choice in
                1) install_wd_cu12x ;;
                2) install_wd_cpu ;;
                *)
                    echo "Invalid option: $cuda_choice"
                    ;;
            esac
            ;;
        2) install_joy ;;
        3) install_huggingface ;;
        4) install_modelscope ;;
        *)
            echo "Invalid option: $choice"
            ;;
    esac
done

echo "Installation completed"