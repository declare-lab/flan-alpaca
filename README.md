```
conda create -n paca python=3.8 -y
conda activate paca
pip install -r requirements.txt
mkdir -p data
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -O data/alpaca.json
```