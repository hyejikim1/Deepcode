# Deepcode

Deepcode: Feedback Codes via Deep Learning, by Hyeji Kim, Yihan Jiang, Sreeram Kannan, Sewoong Oh, and Pramod Viswanath

# Noiseless feedback 

python feedback_main.py -ns -1 : Forward SNR -1dB

python feedback_main.py -ns 0  : Forward SNR  0dB

python feedback_main.py -ns 1  : Forward SNR  1dB

python feedback_main.py -ns 2  : Forward SNR  2dB


# Noisy feedback

python feedback_main.py -ns 0 -fs -3  : Forward SNR 0dB Feedback SNR -3dB

python feedback_main.py -ns 0 -fs 3   : Forward SNR 0dB Feedback SNR  3dB

python feedback_main.py -ns 0 -fs 10  : Forward SNR 0dB Feedback SNR 10dB

python feedback_main.py -ns 0 -fs 20  : Forward SNR 0dB Feedback SNR 20dB


# Normalization layer 

Mean and variance for normalization layer is saved in meanvar/meanvar_Blocklength_FeedbackSNR_ForwardSNR.pickle

meanvar_51_20_-1.pickle  : Blocklength 51 (50 + zero padding) Noiseless Feedback Forward SNR -1dB
meanvar_51_20_0.pickle   : Blocklength 51 (50 + zero padding) Noiseless Feedback Forward SNR  0dB
meanvar_51_20_1.pickle   : Blocklength 51 (50 + zero padding) Noiseless Feedback Forward SNR  1dB
meanvar_51_20_2.pickle   : Blocklength 51 (50 + zero padding) Noiseless Feedback Forward SNR  2dB

meanvar_51_-3.0_0.pickle : Blocklength 51 (50 + zero padding) Feedback SNR -3dB Forward SNR  0dB
meanvar_51_3.0_0.pickle  : Blocklength 51 (50 + zero padding) Feedback SNR  3dB Forward SNR  0dB
meanvar_51_10.0_0.pickle : Blocklength 51 (50 + zero padding) Feedback SNR 10dB Forward SNR  0dB
meanvar_51_20.0_0.pickle : BLocklength 51 (50 + zero padding) Feedback SNR 20dB Forward SNR  0dB

