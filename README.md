# Fungal_Keratitis_Classification

## Installation
```
git clone https://github.com/IGLICT/Fungal_Keratitis_Classification.git
cd Fungal_Keratitis_Classification
pip install -r requirements.txt
```

## Quick Start

Train the stage 1 network

```
python main_stage.py --stage='train' --config='./FungalKeratitis/stage1_swintransformer.yaml'
```

Test the stage 1 network

```
python main_stage.py --stage-'test' --config='./FungalKeratitis/stage1_swintransformer.yaml'
```