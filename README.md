# feedback_code
Discovering Feedback Codes via Deep Learning, by Hyeji Kim, Yihan Jiang, Sreeram Kannan, Sewoong Oh, and Pramod Viswanath

# Noiseless feedback -ns followed by forward SNR

python feedback_main.py -ns -1
python feedback_main.py -ns 0
python feedback_main.py -ns 1
python feedback_main.py -ns 2


# Noisy feedback -ns followed by forward SNR and -fs followed by feedback SNR
python feedback_main.py -ns 0 -fs -3
python feedback_main.py -ns 0 -fs 3
python feedback_main.py -ns 0 -fs 10
python feedback_main.py -ns 0 -fs 20


# Under ./meanvar:  meanvar_A_B_C.pickle holds mean and variance for causal normalization

A: Blocklength (including zero padding; e.g. 51)
B: Feedback SNR (e.g., -3.0, 3.0, 10.0, 20.0 (dB) and 20 for noiseless feedback) 
C: Forward SNR (e.g., -1, 0, 1, 2 dB) 

# Models under ./model
