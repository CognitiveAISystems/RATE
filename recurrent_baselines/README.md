# RNN and SSM Baselines

This directory contains implementations of various RNN and SSM architectures specifically tested on T-Maze and ViZDoom-Two-Colors memory-intensive environments for comparison with our RATE model.

## Implemented Architectures

### DLSTM & DGRU
Decision LSTM (DLSTM) and Decision GRU (DGRU) implementations are based on the [decision-lstm repository](https://github.com/max7born/decision-lstm/). DGRU was derived by adapting the DLSTM architecture to use GRU cells instead of LSTM cells.

### DMamba
Decision Mamba (DMamba) implementation is based on the [decision-mamba repository](https://github.com/Toshihiro-Ota/decision-mamba/tree/main).

## Environments

### T-Maze Environment

> **Note**: All commands must be executed from the repository root directory.

The default configuration uses 3 segments and a context length of 30 (K_{eff} = 90) as specified in `config_lstm.yaml`. Initial experiments with this configuration showed limited success (SR < 50%) for the recurrent baselines.

#### Training Commands

**DLSTM & DGRU:**
```bash
# DLSTM Training
python3 recurrent_baselines/LSTM_GRU/tmaze/lstm_train_tmaze.py \
    --model_mode 'LSTM' \
    --curr 'false' \
    --ckpt_folder 'LSTM_no_curr_max_3' \
    --max_n_final 3 \
    --text 'LSTM_no_curr'

# DGRU Training
python3 recurrent_baselines/LSTM_GRU/tmaze/lstm_train_tmaze.py \
    --model_mode 'GRU' \
    --curr 'false' \
    --ckpt_folder 'GRU_no_curr_max_3' \
    --max_n_final 3 \
    --text 'GRU_no_curr'
```

**DMamba:**
```bash
python3 recurrent_baselines/Mamba/tmaze/mamba_train_tmaze.py \
    --model_mode 'MAMBA' \
    --curr 'false' \
    --ckpt_folder 'MAMBA_max_3_no_curr' \
    --max_n_final 3 \
    --text 'MAMBA_max_3_no_curr'
```

#### Hyperparameter Sweeps

Sweeps are available for different context lengths (K = 9, 30, 90):

**DLSTM & DGRU:**
```bash
# K = 9
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_9/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_9/run_sweep_gru.py

# K = 30
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_30/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_30/run_sweep_gru.py

# K = 90
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_90/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_90/run_sweep_gru.py
```

**DMamba:**
```bash
# K = 9
python3 recurrent_baselines/Mamba/tmaze/sweeps_K_9/run_sweep_mamba.py

# K = 30
python3 recurrent_baselines/Mamba/tmaze/sweeps_K_30/run_sweep_mamba.py

# K = 90
python3 recurrent_baselines/Mamba/tmaze/sweeps_K_90/run_sweep_mamba.py
```

### ViZDoom-Two-Colors Environment

#### Training Commands

**DLSTM & DGRU:**
```bash
# DLSTM Training
python3 recurrent_baselines/LSTM_GRU/vizdoom/lstm_train_vizdoom.py \
    --model_mode 'LSTM' \
    --ckpt_folder 'LSTM' \
    --text 'LSTM'

# DGRU Training
python3 recurrent_baselines/LSTM_GRU/vizdoom/lstm_train_vizdoom.py \
    --model_mode 'GRU' \
    --ckpt_folder 'GRU' \
    --text 'GRU'
```

**DMamba:**
```bash
python3 recurrent_baselines/Mamba/vizdoom/mamba_train_vizdoom.py \
    --model_mode 'MAMBA' \
    --ckpt_folder 'MAMBA' \
    --text 'MAMBA'
```

#### Hyperparameter Sweeps

**DLSTM & DGRU:**
```bash
python3 recurrent_baselines/LSTM_GRU/vizdoom/sweeps/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/vizdoom/sweeps/run_sweep_gru.py
```

**DMamba:**
```bash
python3 recurrent_baselines/Mamba/vizdoom/sweeps/run_sweep_mamba.py
```

## Sweep Results

### GRU K = 9
![image info](./assets/gru_k9.png)

### LSTM K = 9
![image info](./assets/lstm_k9.png)

### MAMBA K = 9
![image info](./assets/mamba_k9.png)

### GRU K = 30
![image info](./assets/gru_k30.png)

### LSTM K = 30
![image info](./assets/lstm_k30.png)

### MAMBA K = 30
![image info](./assets/mamba_k30.png)

### GRU K = 90
![image info](./assets/gru_k90.png)

### LSTM K = 90
![image info](./assets/lstm_k90.png)

### MAMBA K = 90
![image info](./assets/mamba_k90.png)