#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.ssh
chmod 700 /root/.ssh

# copy only what you need
for f in config id_rsa id_rsa.pub id_ed25519 id_ed25519.pub known_hosts; do
  if [ -f "/mnt/host-ssh/$f" ]; then
    cp "/mnt/host-ssh/$f" "/root/.ssh/$f"
  fi
done

chmod 600 /root/.ssh/config 2>/dev/null || true
chmod 600 /root/.ssh/id_* 2>/dev/null || true
chmod 644 /root/.ssh/*.pub /root/.ssh/known_hosts 2>/dev/null || true
