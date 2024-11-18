# DLSTM & DGRU

Decision LSTM (DLSTM) code was adapted for `T-Maze` and `ViZDoom-Two-Colors` from https://github.com/max7born/decision-lstm/, and Decision GRU (DGRU) was obtained by replacing LSTM with GRU in the DLSTM code.

## T-Maze

All code must be run from the root directory of the repository.

By default, we consider 3 segments and a context length of 30 (K_{eff} = 90) (see `config_lstm.yaml`) to compare DLSTM and DGRU against RATE. However, after conducting a sweep with this effective context configuration, we were unable to achieve SR > 50% for these recurrent baselines. 

To run training, use the following commands:
```bash
python3 recurrent_baselines/LSTM_GRU/tmaze/lstm_train_tmaze.py --model_mode 'LSTM' --curr 'false' --ckpt_folder 'LSTM_no_curr_max_3' --max_n_final 3 --text 'LSTM_no_curr'

python3 recurrent_baselines/LSTM_GRU/tmaze/lstm_train_tmaze.py --model_mode 'GRU' --curr 'false' --ckpt_folder 'GRU_no_curr_max_3' --max_n_final 3 --text 'GRU_no_curr'
```

To run wandb sweeps, use the following commands (for K = 9, 30, 90):
```bash
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_9/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_9/run_sweep_gru.py

python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_30/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_30/run_sweep_gru.py

python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_90/run_sweep_lstm.py
python3 recurrent_baselines/LSTM_GRU/tmaze/sweeps_K_90/run_sweep_gru.py
```