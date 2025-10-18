set -ex
apt update
apt install -y lshw
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

