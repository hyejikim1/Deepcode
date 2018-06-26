# Noiseless feedback

python feedback_main.py -ns -1
python feedback_main.py -ns 0
python feedback_main.py -ns 1
python feedback_main.py -ns 2


# Noisy feedback 
python feedback_main.py -ns 0 -fs -3
python feedback_main.py -ns 0 -fs 3
python feedback_main.py -ns 0 -fs 10
python feedback_main.py -ns 0 -fs 20



# Load saved mean and variance for causal normalization

meanvar/meanvar_A_B_C.pickle

A: Blocklength (including zero padding; e.g. 51)
B: Feedback SNR (e.g., -3.0, 3.0, 10.0, 20.0 (dB) and 20 for noiseless feedback) 
C: Forward SNR (e.g., -1, 0, 1, 2 dB) 

# Saved models
