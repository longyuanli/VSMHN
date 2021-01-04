## Codebase for "Synergetic Learning of Heterogeneous Temporal Sequences for Multi-Horizon Probabilistic Forecasting"

### Code explanation

1. data_process.py
   - Transform raw time-series data to preprocessed time-series data
   - Extract events from variables
   - Create formatted data for training
2. dataset.py
   - Dataset class for heterogeneous sequences
3. evaluation.py
   - Functions for evaluation of forecasting resutls
4. VSMHN.py
   - Implementation codes of the proposed model
5. run.py
   - Runner 
6. utils.py
   - some useful utility functions

### Dependencies

```
properscoring==0.1
scipy==1.2.1
torch==1.3.0
numpy==1.18.1
requests==2.21.0
tqdm==4.36.1
pandas==0.25.3
matplotlib==3.0.3
scikit_learn==0.23.2
```

### Requirements

```
pip install -r requirements.txt
```

### Command Inputs

```
usage: run.py [-h] [--X_context X_CONTEXT] [--y_horizon Y_HORIZON]
              [--window_skip WINDOW_SKIP] [--train_prop TRAIN_PROP]
              [--h_dim H_DIM] [--z_dim Z_DIM] [--use_GRU USE_GRU] [--lr LR]
              [--dec_bound DEC_BOUND] [--batch_size BATCH_SIZE]
              [--epochs EPOCHS] [--device DEVICE] [--mc_times MC_TIMES]

optional arguments:
  -h, --help            show this help message and exit
  --X_context X_CONTEXT
                        observing time length (default: 168)
  --y_horizon Y_HORIZON
                        predicting time length (default: 24)
  --window_skip WINDOW_SKIP
                        skipping step for data generation (default: 12)
  --train_prop TRAIN_PROP
                        percent of data used for trainning (default: 0.97)
  --h_dim H_DIM         dimension for ts/event encoder and decoder (default:
                        200)
  --z_dim Z_DIM         dimension for latent variable encoder (default: 100)
  --use_GRU USE_GRU     RNN cell type(True:GRU, False:LSTM) (default: True)
  --lr LR               learning_rate (default: 0.001)
  --dec_bound DEC_BOUND
                        dec_bound for std (default: 0.05)
  --batch_size BATCH_SIZE
                        batch size (default: 400)
  --epochs EPOCHS       trainning epochs (default: 100)
  --device DEVICE       select device (cuda:0, cpu) (default: cuda:0)
  --mc_times MC_TIMES   num of monte carlo simulations (default: 1000)
```

### Example command

```
$ python run.py --h_dim 200 --z_dim 100 --batch_size 400 --epochs 100 --device cuda:0
```

### Outputs

- CRPS score and RMSE score of forecast variables
- forecast_plots.png visualizes forecast results

