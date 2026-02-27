set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y --no-install-recommends \
  software-properties-common \
  wget curl git openssh-client \
  python3-dev python3-pip python3-wheel python3-setuptools \
  zlib1g g++ freeglut3-dev \
  libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev

pip3 cache purge || true
apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*