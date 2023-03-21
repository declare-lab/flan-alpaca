```
conda create -n paca python=3.8 -y
conda activate paca
pip install -r requirements.txt
mkdir -p data
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -O data/alpaca.json
wget https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned.json -O data/alpaca_clean.json
```