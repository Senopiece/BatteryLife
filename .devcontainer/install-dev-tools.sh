# update system
apt-get update
apt-get upgrade -y

# install Linux tools and Python 3
apt-get install software-properties-common wget curl git \
    python3-dev python3-pip python3-wheel python3-setuptools -y

# install Python packages
python3 -m pip install --upgrade pip
pip3 install -r .devcontainer/requirements.txt

# update CUDA Linux GPG repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb

# install cuDNN for CUDA 12.1
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-get update
apt-get install -y \
  libcudnn8=8.9.7.29-1+cuda12.1 \
  libcudnn8-dev=8.9.7.29-1+cuda12.1

# install exact NVIDIA driver libraries to match host kernel driver 570.133.20
apt-get install -y \
  libnvidia-compute-570=570.133.20-0ubuntu1 \
  libnvidia-decode-570=570.133.20-0ubuntu1 \
  nvidia-utils-570=570.133.20-0ubuntu1

# Remove duplicate CUDA source-list (silences apt warning)
rm /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list
apt-get update

# install recommended packages
apt-get install zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y

# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean