__author__ = 'hyejikim'

# Deepcode simulation. 
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras import regularizers
from keras.engine.topology import Layer
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.engine import Layer
import scipy.io as sio
import matplotlib, h5py, pickle, sys, time


################################
# GPU memory allocation 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
print '[Test][Warining] Restrict GPU memory usage to 60%'
 
################################
# Arguments
n_inp = sys.argv[1:]

if '-coderate' in n_inp:
    ind1 = n_inp.index('-coderate')
    coderate = int(n_inp[ind1+1])
else:
    coderate = 3
 
if '-tx' in n_inp:
    ind1 = n_inp.index('-tx')
    num_hunit_rnn_tx = int(n_inp[ind1+1])
else:
    num_hunit_rnn_tx = 50
if '-rx' in n_inp:
    ind1 = n_inp.index('-rx')
    num_hunit_rnn_rx = int(n_inp[ind1+1])
else:
    num_hunit_rnn_rx = 50 
print 'Tx hidden nodes:', num_hunit_rnn_tx
print 'Rx hidden nodes:', num_hunit_rnn_rx


if '-len' in n_inp:
    ind1      = n_inp.index('-len')
    bit_length = int(n_inp[ind1+1])
else:
    bit_length = 51 # Number of bits including one (for zero padding)
print 'Block length: ', bit_length

if '-fs' in n_inp: # Noisy feedback
    ind1      = n_inp.index('-fs')
    fsSNR = float(n_inp[ind1+1])
    feedback_sigma = 10**(-fsSNR*1.0/20)
else:
    fsSNR = 20 # fsSNR = 20 means noiseless feedback
    feedback_sigma = 0

if '-ns' in n_inp:
    ind1      = n_inp.index('-ns')
    nsSNR = int(n_inp[ind1+1])
    noise_sigma = 10**(-nsSNR*1.0/20)
else:
    nsSNR = 0
    noise_sigma = 10**(-nsSNR*1.0/20)
print 'SNR of forward channel: ', nsSNR
print 'SNR of feedback channel: ', fsSNR


if '-k' in n_inp:
	ind1 = n_inp.index('-k')
	k = int(n_inp[ind1+1])
else: 
        k = bit_length*200000 # length of total message bits for testing.
print 'Total number of bits for testing: ', k

if '-noncausal' in n_inp:
    causal = False
else:
    causal = True        
print 'Causality: ', causal

 
class ScaledLayer(Layer): # Power Allocation Layer
    def __init__(self, **kwargs):
        super(ScaledLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.output_dim = input_shape[1] 
        self.W = self.add_weight(name = 'power_weight', shape=(1,), # Power allocation for information bit stream
                                 initializer='ones', trainable=True)
        self.W2 = self.add_weight(name = 'power_weight1', shape=(1,), # Power allocation for parity 1 stream
                                 initializer='ones', trainable=True)
        self.W3 = self.add_weight(name = 'power_weight2', shape=(1,), # Power allocation for parity 2 stream
                                 initializer='ones', trainable=True)
         
        self.b1 = self.add_weight(name = 'b1', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for last-4 bit
        self.b2 = self.add_weight(name = 'b2', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for last-3 bit
        self.b3 = self.add_weight(name = 'b3', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for last-2 bit
        self.b4 = self.add_weight(name = 'b4', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for last-1 bit
        self.b5 = self.add_weight(name = 'b5', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for last bit
  
        self.g1 = self.add_weight(name = 'g1', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 1st bit
        self.g2 = self.add_weight(name = 'g2', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 2nd bit
        self.g3 = self.add_weight(name = 'g3', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 3rd bit
        self.g4 = self.add_weight(name = 'g4', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 4th bit
 

        super(ScaledLayer, self).build(input_shape)
    def call(self, x, mask=None):
        sys = tf.reshape(tf.multiply(x[:,:,0], self.W),[tf.shape(x)[0],tf.shape(x)[1],1])
        par1 = tf.reshape(tf.multiply(x[:,:,1], self.W2),[tf.shape(x)[0],tf.shape(x)[1],1])
        par2 = tf.reshape(tf.multiply(x[:,:,2], self.W3),[tf.shape(x)[0],tf.shape(x)[1],1])
 
        cats = K.concatenate([K.concatenate([tf.expand_dims(tf.multiply(self.g1,sys[:,0,:]),1),tf.expand_dims(tf.multiply(self.g1,par1[:,0,:]),1),tf.expand_dims(tf.multiply(self.g1,par2[:,0,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.g2,sys[:,1,:]),1),tf.expand_dims(tf.multiply(self.g2,par1[:,1,:]),1),tf.expand_dims(tf.multiply(self.g2,par2[:,1,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.g3,sys[:,2,:]),1),tf.expand_dims(tf.multiply(self.g3,par1[:,2,:]),1),tf.expand_dims(tf.multiply(self.g3,par2[:,2,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.g4,sys[:,3,:]),1),tf.expand_dims(tf.multiply(self.g4,par1[:,3,:]),1),tf.expand_dims(tf.multiply(self.g4,par2[:,3,:]),1)],axis=2),
                              K.concatenate([sys[:,4:bit_length-5,:],par1[:,4:bit_length-5,:],par2[:,4:bit_length-5,:]],axis=2),                              
                              K.concatenate([tf.expand_dims(tf.multiply(self.b1,sys[:,bit_length-5,:]),1),tf.expand_dims(tf.multiply(self.b1,par1[:,bit_length-5,:]),1),tf.expand_dims(tf.multiply(self.b1,par2[:,bit_length-5,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.b2,sys[:,bit_length-4,:]),1),tf.expand_dims(tf.multiply(self.b2,par1[:,bit_length-4,:]),1),tf.expand_dims(tf.multiply(self.b2,par2[:,bit_length-4,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.b3,sys[:,bit_length-3,:]),1),tf.expand_dims(tf.multiply(self.b3,par1[:,bit_length-3,:]),1),tf.expand_dims(tf.multiply(self.b3,par2[:,bit_length-3,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.b4,sys[:,bit_length-2,:]),1),tf.expand_dims(tf.multiply(self.b4,par1[:,bit_length-2,:]),1),tf.expand_dims(tf.multiply(self.b4,par2[:,bit_length-2,:]),1)],axis=2),
                              K.concatenate([tf.expand_dims(tf.multiply(self.b5,sys[:,bit_length-1,:]),1),tf.expand_dims(tf.multiply(self.b5,par1[:,bit_length-1,:]),1),tf.expand_dims(tf.multiply(self.b5,par2[:,bit_length-1,:]),1)],axis=2),
                            ], axis=1)
 
        cats_mean, cats_var = tf.nn.moments(cats,[0])
        rem = bit_length-9.0
 
        adj = bit_length*1.0/(bit_length)
        den = (rem + self.g1**2 + self.g2**2 + self.g3**2 + self.g4**2 + self.b1**2 + self.b2**2 + self.b3**2 + self.b4**2 + self.b5**2)*(self.W**2+self.W2**2+self.W3**2)
        return tf.sqrt(3.0*bit_length/den)*cats

    def get_output_shape_for(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])
 
    def compute_output_shape(self, input_shape):
        a_shape = input_shape
        return (a_shape[0], a_shape[1], a_shape[2])
 
 
# Encoder. Single Directional. One layer RNN
f1 = SimpleRNN(name='simple_rnn_1', units=num_hunit_rnn_tx, activation='tanh', return_sequences=True, dropout=1.0)
f3 = TimeDistributed(Dense(coderate-1, activation='sigmoid'),name = 'time_distributed_0')
 
# Decoder. Bidirectional. Two Layered GRU with batch normalization. 
f4 = Bidirectional(GRU(name='bidirectional_1', units=num_hunit_rnn_rx, activation='tanh', return_sequences=True, dropout=1.0))
f5 = BatchNormalization(name='batch_normalization_1')
f6 = Bidirectional(GRU(name='bidirectional_2', units=num_hunit_rnn_rx, activation='tanh', return_sequences=True, dropout=1.0))
f7 = BatchNormalization(name='batch_normalization_2') 
f8 = TimeDistributed(Dense(1, activation='sigmoid'), name='time_distributed_1')
 

# Loss used for training: Binary crossentropy over all bits except for the zero padding
def customLoss(y_true,y_pred):
    y_true_50 = y_true[:,0:bit_length-1,:]
    y_pred_50 = y_pred[:,0:bit_length-1,:]     
    return K.binary_crossentropy(y_true_50, y_pred_50)

# Errors used for training: ignoring the error on the zero padded bits
def errors(y_true, y_pred):
    y_true_50 = y_true[:,0:bit_length-1,:]
    y_pred_50 = y_pred[:,0:bit_length-1,:]
    myOtherTensor = K.not_equal(y_true_50, K.round(y_pred_50))
    return K.mean(tf.cast(myOtherTensor, tf.float32))
 
# Normalization layer of the encoder
def normalize(x):
    if causal == False: # Average over batches
        x_mean, x_var = tf.nn.moments(x,[0])
    else: # Load pre-computed mean/variance for normalization 
        id = str(bit_length)+'_'+str(fsSNR)+'_'+str(nsSNR)
        with open('meanvar/meanvar_'+id+'.pickle') as g:  # Python 3: open(..., 'wb')
            mean1, var1 = pickle.load(g)
        x_mean = tf.Variable(mean1, tf.float32)
        x_var = tf.Variable(var1, tf.float32)
       
    x = (x-x_mean)*1.0/tf.sqrt(x_var)
    return x
 
# coderate. takeNoise
def takeNoise(x):
    return tf.reshape(x[:,:,coderate+1:2*coderate+1],[tf.shape(x[:,:,0])[0],bit_length,coderate]) # 4 - N_i // 5 - M_i // 6 - O_i  
# takeBit. BPSK modulation
def takeBit(x):
    return tf.reshape(2*x[:,:,0]-1,[tf.shape(x[:,:,0])[0],bit_length,1])

def concat(x):
    return K.concatenate(x)
 
inputs = Input(shape=(bit_length, 2*coderate+1))
x = inputs

# Take input for parity generation
def split_data_input_noisedelay(x):
    x1 = x[:,:,0:coderate+1] # E.g., for coderate=3: 0 - b_i // 1 - N_i in Phase I // 2 - M_{i-1} in Phase II // 3 - O_{i-1} in Phase II.
    return x1
 
parity = f3(f1(Lambda(split_data_input_noisedelay)(x))) # Generate parity based on message bits and Phase I noise and delayed Phase II noise
norm_parity = Lambda(normalize)(parity) # Normalize the parity
codeword = Lambda(concat)([Lambda(takeBit)(x),norm_parity]) # Codeword: raw bits and normalized parity
powerd_codeword = ScaledLayer(name='noload_abr')(codeword) # Codeword after Power Allocation

noise = Lambda(takeNoise)(x)
noisy_received = keras.layers.add([powerd_codeword,noise]) # Received value: Sum of noise & codeword 
predictions = f8(f7(f6(f5(f4(noisy_received))))) # Decoder output
 
# output of model_cw is encoder's power allocated codeword
model_cw = Model(inputs=inputs, outputs=powerd_codeword)
optimizer= keras.optimizers.adam(lr=0.02,clipnorm=1.)
model_cw.compile(optimizer=optimizer,loss=customLoss, metrics=[errors])
 
# output of model is decoder's estimate
model = Model(inputs=inputs, outputs=predictions)
optimizer= keras.optimizers.adam(lr=0.02,clipnorm=1.)
model.compile(optimizer=optimizer,loss=customLoss, metrics=[errors])

id = str(bit_length)+'_'+str(fsSNR)+'_'+str(nsSNR)
 
# Load model
if '-fs' in n_inp:
    model.load_weights('model/round3_powerabr_new_noisy_nettype_rnnrate3tx_50_rx_50_len_51_'+str(fsSNR)+'_0.h5',by_name=True)
    print 'model noise', str(fsSNR),'dB'
else:
    model.load_weights('model/round4_powerabr_new_nettype_rnnrate3tx_50_rx_50_len_51_20_'+str(nsSNR)+'.h5',by_name=True)
    print 'model', str(nsSNR),'dB'
    

# Generate test examples: X_train (X_test) is true label. X_train_noise (X_test_noise) is input to the neural network
# Generate test examples: information bits X_train (X_test)
print 'Generate test examples'
X_train_raw = np.random.randint(0,2,k)
X_test_raw  = np.random.randint(0,2,k)
X_train = X_train_raw.reshape((k/bit_length, bit_length, 1))
X_test  = X_test_raw.reshape((k/bit_length, bit_length, 1))

# Generate test examples: input to the neural network X_train_noise (X_test_noise)
# Input to neural network: message bits and noise sequence in Phase I(n_1,...,n_bitlength) and Phase II (m_1, o_1, m_2, o_2, ..., m_bitlength, o_bitlength)
# Form the input as: X_train_noise[batch_index,i,:] = [b_i, n_i, m_{i-1}, o_{i-1}, n_i, m_i, o_i] for i = 1:bitlength

X_train_noise = np.zeros([k/bit_length, bit_length, 2*coderate+1])
X_train_noise[:,:,0] = X_train[:,:,0] # True message bits
X_train_noise[:,bit_length-1,0] = np.zeros(X_train_noise[:,bit_length-1,0].shape) # Set the last Bit to be 0.

for inx in range(1,coderate+1):
    X_train_noise[:,:,coderate+inx] = noise_sigma * np.random.standard_normal(X_train_noise[:,:,coderate+inx].shape) # Noise
    if inx == 1:
        X_train_noise[:,:,inx] = np.roll(X_train_noise[:,:,coderate+inx], 0, axis=1) + feedback_sigma * np.random.standard_normal(X_train_noise[:,:,3].shape)  # Delayed Noise
    else:
        X_train_noise[:,:,inx] = np.roll(X_train_noise[:,:,coderate+inx], 1, axis=1) + feedback_sigma * np.random.standard_normal(X_train_noise[:,:,4].shape)  # Delayed Noise
        X_train_noise[:,0,inx] = 0

X_test_noise = np.zeros([k/bit_length, bit_length,2*coderate+1])
X_test_noise[:,:,0] = X_test[:,:,0] # True message bits
X_test_noise[:,bit_length-1,0] = np.zeros(X_test_noise[:,bit_length-1,0].shape) # Set the last Bit to be 0. 
for inx in range(1,coderate+1):
    X_test_noise[:,:,coderate+inx] = noise_sigma * np.random.standard_normal(X_test_noise[:,:,coderate+inx].shape) # Noise
    if inx == 1:
        X_test_noise[:,:,inx] = np.roll(X_test_noise[:,:,coderate+inx], 0, axis=1) + feedback_sigma * np.random.standard_normal(X_test_noise[:,:,3].shape)  # Delayed Noise
    else:
        X_test_noise[:,:,inx] = np.roll(X_test_noise[:,:,coderate+inx], 1, axis=1) + feedback_sigma * np.random.standard_normal(X_test_noise[:,:,4].shape)  # Delayed Noise
        X_test_noise[:,0,inx] = 0


print '-------Evaluation start-------'
test_batch_size = 200
codewords = model_cw.predict(X_test_noise, batch_size=test_batch_size)
print 'power of codewords: ', np.var(codewords)
print 'mean of codewords: ', np.mean(codewords)

predicted = np.round(model.predict(X_test_noise, batch_size=test_batch_size))
predicted = predicted[:,0:bit_length-1,:] # Ignore the last bit (zero padding) 
target = X_test[:,0:bit_length-1,:].reshape([X_test.shape[0],X_test.shape[1]-1,1]) # Ignore the last bit (zero padding)

# BER
c_ber = 1- sum(sum(predicted == target))*\
       1.0/(target.shape[0] * target.shape[1] *target.shape[2])
# BLER
tp0 = (abs(np.round(predicted)-target)).reshape([target.shape[0],target.shape[1]]) 
bler = sum(np.sum(tp0,axis=1)>0)*1.0/(target.shape[0])

print 'BER of decoder estimate: ', c_ber[0]
print 'BLER of decoder estimate: ', bler

# Interpret: generate Figure 5
interpret = True

if interpret == True:

    r1 = X_test_noise[:,:,0] # b_i
    n1 = X_test_noise[:,:,1] # N_i
    n2 = X_test_noise[:,:,2] # M_{i-1}
    n3 = X_test_noise[:,:,3] # O_{i-1}
    p1 = codewords[:,:,1]    # Parity1_i
    p2 = codewords[:,:,2]    # Parity2_i 
    
    num_sample_points = 20 # Number of sample points
    rr1 = r1[0:num_sample_points,:] # b_i
    nn1 = n1[0:num_sample_points,:] # N_i
    nn2 = n2[0:num_sample_points,:] # M_{i-1}
    nn3 = n3[0:num_sample_points,:] # O_{i-1}
    pp1 = p1[0:num_sample_points,:] # Parity1_i
    pp2 = p2[0:num_sample_points,:] # Parity2_i

    plt.close()
    plt.plot(nn1[rr1==0],pp1[rr1==0],'r.')
    plt.plot(nn1[rr1==1],pp1[rr1==1],'bx')
    plt.savefig('figs/SNR'+str(nsSNR)+'plot'+str(num_sample_points)+'_PhaseI_noise_vs_parity1.png')

    plt.close()
    plt.plot(nn1[rr1==0],pp2[rr1==0],'r.')
    plt.plot(nn1[rr1==1],pp2[rr1==1],'bx')
    plt.savefig('figs/SNR'+str(nsSNR)+'plot'+str(num_sample_points)+'_PhaseI_noise_vs_parity2.png')

    plt.close()
    plt.plot(pp1[rr1==0],pp2[rr1==0],'r.')
    plt.plot(pp1[rr1==1],pp2[rr1==1],'bx')
    plt.savefig('figs/SNR'+str(nsSNR)+'plot_'+str(num_sample_points)+'_parity1_vs_parity2.png')





