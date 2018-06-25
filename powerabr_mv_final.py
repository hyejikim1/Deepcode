# Feedback code simulation. 
# Load presaved mean and variance! 
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.engine.topology import Layer
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.engine import Layer
import scipy.io as sio
import matplotlib
import h5py
import pickle
import sys
import time


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

if '-lr' in n_inp:
    ind1 = n_inp.index('-lr')
    learning_rate = float(n_inp[ind1+1])

 
if '-coderate' in n_inp:
    ind1 = n_inp.index('-coderate')
    coderate = int(n_inp[ind1+1])
else:
    coderate = 3
  
nettype = 'rnn'
 
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
 
if '-howmany' in n_inp:
    ind1 = n_inp.index('-howmany')
    howmany = int(n_inp[ind1+1])
else:
    howmany = 10

if '-len' in n_inp:
    ind1      = n_inp.index('-len')
    bit_length = int(n_inp[ind1+1])
else:
    bit_length = 51 # Number of bits including one (for zero padding)
    print bit_length

# Whether to run Understanding code (us.py)
run_us = True
learning_rate = 0.02

 
if '-fs' in n_inp:
    ind1      = n_inp.index('-fs')
    fsSNR = float(n_inp[ind1+1])
else:
    fsSNR = 20

if fsSNR == 20: # fsSNR = 20 means noiseless feedback
    feedback_sigma = 0
else:
    feedback_sigma = 10**(-fsSNR*1.0/20)


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
    
 
class ScaledLayer(Layer): # a scaled layer
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
                                 initializer='ones', trainable=True) # Power allocation for 1st bit
        self.b2 = self.add_weight(name = 'b2', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 2nd bit
        self.b3 = self.add_weight(name = 'b3', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 3rd bit
        self.b4 = self.add_weight(name = 'b4', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 4th bit
        self.b5 = self.add_weight(name = 'b5', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for 5th bit
 
        self.g1 = self.add_weight(name = 'g1', shape=(1,), 
                                 initializer='ones', trainable=True) # Power allocation for bit_length - 
        self.g2 = self.add_weight(name = 'g2', shape=(1,), 
                                 initializer='ones', trainable=True)
        self.g3 = self.add_weight(name = 'g3', shape=(1,), 
                                 initializer='ones', trainable=True)
        self.g4 = self.add_weight(name = 'g4', shape=(1,), 
                                 initializer='ones', trainable=True)
 
 
        super(ScaledLayer, self).build(input_shape)  # Be sure to call this somewhere!
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
        print cats_mean.shape
        print cats_var.shape
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
 

# Setup LR decay
def scheduler(epoch):
 
    if epoch > 2 and epoch <=3:
        print 'changing by /10 lr'
        lr = learning_rate/10.0
    elif epoch >3 and epoch <=5:
        print 'changing by /100 lr'
        lr = learning_rate/100.0
    elif epoch >5 and epoch <=7:
        print 'changing by /1000 lr'
        lr = learning_rate/1000.0
    elif epoch > 7:
        print 'changing by /10000 lr'
        lr = learning_rate/10000.0
    else:
        lr = learning_rate
 
    return lr
 
change_lr = LearningRateScheduler(scheduler)
 
print 'Tx hidden nodes:', num_hunit_rnn_tx
print 'Rx hidden nodes:', num_hunit_rnn_rx
 
# Encoder. Single Directional. One layer RNN
f1 = SimpleRNN(name='simple_rnn_1', units=num_hunit_rnn_tx, activation='tanh', return_sequences=True, dropout=1.0)
f3 = TimeDistributed(Dense(coderate-1, activation='sigmoid'),name = 'time_distributed_0')
 
# Decoder. Bidirectional. Two Layered GRU with batch normalization. 
f4 = Bidirectional(GRU(name='bidirectional_1', units=num_hunit_rnn_rx, activation='tanh', return_sequences=True, dropout=1.0))
f5 = BatchNormalization(name='batch_normalization_1')
f6 = Bidirectional(GRU(name='bidirectional_2', units=num_hunit_rnn_rx, activation='tanh', return_sequences=True, dropout=1.0))
f7 = BatchNormalization(name='batch_normalization_2') 
f8 = TimeDistributed(Dense(1, activation='sigmoid'), name='time_distributed_1')
 

# Errors used for training: ignoring the error on the zero padded bits
def errors(y_true, y_pred):
    y_true_50 = y_true[:,0:bit_length-1,:]
    y_pred_50 = y_pred[:,0:bit_length-1,:]
 
    myOtherTensor = K.not_equal(y_true_50, K.round(y_pred_50))
    return K.mean(tf.cast(myOtherTensor, tf.float32))
 

if '-causal' in n_inp:
    ind1 = n_inp.index('-causal')
    causal = True
else:
    causal = True
 
def normalize(x):
    if causal == False:
        x_mean, x_var = tf.nn.moments(x,[0])
    else:
        id = str(bit_length)+'_'+str(fsSNR)+'_'+str(nsSNR)
	if '-pid' in n_inp:
		ind1 = n_inp.index('-pid')
		id = str(n_inp[ind1+1])
		print 'str(bit_length)+_+str(fsSNR)+_+str(nsSNR) is ', id
        with open('meanvar_'+id+'.pickle') as g:  # Python 3: open(..., 'wb')
            mean1, var1 = pickle.load(g)

        x_mean = tf.Variable(mean1, tf.float32)
        x_var = tf.Variable(var1, tf.float32)
       
    x = (x-x_mean)*1.0/tf.sqrt(x_var)
    return x
 
def concat0(x):
    #print tf.shape(x)
    padding = tf.zeros([tf.shape(x)[0],tf.shape(x)[1],1], tf.float32)
    return K.concatenate([x, padding])
    #return K.concatenate([x,tf.cast(tf.zeros(tf.shape(x)[0],tf.shape(x)[1],1),tf.float32)])
    #return tf.cast(tf.zeros(bit_length,1),tf.float32)
 
def concat(x):
    return K.concatenate(x)
 
 
def sum2(x):
    return tf.reshape(x[:,:,0]+x[:,:,1],[tf.shape(x[:,:,0])[0],bit_length,1])
 
# coderate. takeNoise
def takeNoise(x):
    return tf.reshape(x[:,:,coderate+1:2*coderate+1],[tf.shape(x[:,:,0])[0],bit_length,coderate]) # 3 - noise1. 4 - noise2. 
 
# takeBit. always the same
def takeBit(x):
    return tf.reshape(2*x[:,:,0]-1,[tf.shape(x[:,:,0])[0],bit_length,1])
 
 
 
delay_array = np.array(range(0,bit_length))
 
 
inputs = Input(shape=(bit_length, 2*coderate+1))
 
 
x = inputs
 
# Easiest Rate 1/4. Generate all three parities together...
def split_data_input_noisedelay(x):
    x1 = x[:,:,0:coderate+1] # 0 - bits. 1 - noise1. 2 - noise2.
    return x1
 
parity = f3(f1(Lambda(split_data_input_noisedelay)(x)))
norm_parity = Lambda(normalize)(parity)
 
codeword = Lambda(concat)([Lambda(takeBit)(x),norm_parity])
 
powerd_codeword = ScaledLayer(name='noload_abr')(codeword)
 
 
 
 
noise = Lambda(takeNoise)(x) # Take Noise
 
noisy_received = keras.layers.add([powerd_codeword,noise]) # Sum Noise and the Codeword
 
 
#noisy_received = Lambda(concat)([norm_codeword, noise])
#noisy_received = sum2(noisy_received)
 
'''
print noisy_received.shape[0]
print noisy_received.shape[1]
print noisy_received.shape[2]
'''
 
x = noisy_received#Lambda(concat0)(noisy_received)
x = f8(f7(f6(f5(f4(x)))))
 
predictions  = x
 
 
 
 
def customLoss(y_true,y_pred):
    y_true_50 = y_true[:,0:bit_length-1,:]
    y_pred_50 = y_pred[:,0:bit_length-1,:]
     
    return K.binary_crossentropy(y_true_50, y_pred_50) #K.sum(K.log(yTrue) - K.log(yPred))
 
 
model0 = Model(inputs=inputs, outputs=parity)
optimizer= keras.optimizers.adam(lr=learning_rate,clipnorm=1.)
model0.compile(optimizer=optimizer,loss=customLoss, metrics=[errors])
 
 
model_cw = Model(inputs=inputs, outputs=powerd_codeword)
optimizer= keras.optimizers.adam(lr=learning_rate,clipnorm=1.)
model_cw.compile(optimizer=optimizer,loss=customLoss, metrics=[errors])
 
 
 
model = Model(inputs=inputs, outputs=predictions)
optimizer= keras.optimizers.adam(lr=learning_rate,clipnorm=1.)
model.compile(optimizer=optimizer,loss=customLoss, metrics=[errors])

 
id = str(bit_length)+'_'+str(fsSNR)+'_'+str(nsSNR)
 
# Load!
 
if nsSNR == -2:
    model.load_weights('power_nettype_rnnrate3tx_50_rx_50_len_101_20_-2.0_0.523126187246.h5',by_name=True)
    print 'Load for -2dB power'
elif nsSNR == -1:
    #model.load_weights('power_nettype_rnnrate3tx_50_rx_50_len_101_20_-1.0_0.906540094873.h5',by_name=True)
    #print 'Load for -1dB power'
    model.load_weights('round4_powerabr_new_nettype_rnnrate3tx_50_rx_50_len_'+str(501)+'_20_-1.h5')
    print 'Load previous -1dB length'
    
elif nsSNR == 0:
    model.load_weights('power_nettype_rnnrate3tx_50_rx_50_len_101_20_0.0_0.116625772133.h5',by_name=True)
    print 'load for 0dB power'
elif nsSNR == 1:
    model.load_weights('power_nettype_rnnrate3tx_50_rx_50_len_101_20_0.0_0.940006580573.h5',by_name=True)
    print 'load for 1dB power'
elif nsSNR == 2:
    model.load_weights('power_nettype_rnnrate3tx_50_rx_50_len_101_20_2.0_0.605843714412.h5',by_name=True)
    print 'load for 2dB power'
elif nsSNR == -6:
    model.load_weights('power_nettype_rnnrate3tx_50_rx_50_len_101_20_-6.0_0.938250914242.h5',by_name=True)
    print 'load for -6dB power'

    
# Across round. 
berss = []
blerss = []
 
## TEST

print 'k is ', k
if '-fs' in n_inp:
    model.load_weights('round3_powerabr_new_noisy_nettype_rnnrate3tx_50_rx_50_len_51_'+str(fsSNR)+'_0.h5')
    print 'model noise', str(fsSNR),'dB'
else:
    model.load_weights('round4_powerabr_new_nettype_rnnrate3tx_50_rx_50_len_51_20_'+str(nsSNR)+'.h5')
    print 'model', str(nsSNR),'dB'
    

print 'Generate test examples'
# Generate random bits
X_train_raw = np.random.randint(0,2,k)
X_test_raw  = np.random.randint(0,2,k)
X_train = X_train_raw.reshape((k/bit_length, bit_length, 1))
X_test  = X_test_raw.reshape((k/bit_length, bit_length, 1))

# Input to neural network: message bits and noise sequence in Phase I(n_1,...,n_bitlength) and Phase II (m_1, o_1, m_2, o_2, ..., m_bitlength, o_bitlength)
# Form the input as: X_train_noise[batch_index,i,:] = [b_i, n_i, m_{i-1}, o_{i-1}, n_i, m_i, o_i] for i = 1:bitlength 

X_train_noise = np.zeros([k/bit_length, bit_length, 2*coderate+1])
X_train_noise[:,:,0] = X_train[:,:,0] # True message bits
X_train_noise[:,bit_length-1,0] = np.zeros(X_train_noise[:,bit_length-1,0].shape) # Set Last Bit to be 0.

for inx in range(1,coderate+1):#HERE. BELOW.
    X_train_noise[:,:,coderate+inx] = noise_sigma * np.random.standard_normal(X_train_noise[:,:,coderate+inx].shape) # Noise
    if inx == 1:
        X_train_noise[:,:,inx] = np.roll(X_train_noise[:,:,coderate+inx], 0, axis=1) + feedback_sigma * np.random.standard_normal(X_train_noise[:,:,3].shape)  # Delayed Noise
    else:
        X_train_noise[:,:,inx] = np.roll(X_train_noise[:,:,coderate+inx], 1, axis=1) + feedback_sigma * np.random.standard_normal(X_train_noise[:,:,4].shape)  # Delayed Noise
        X_train_noise[:,0,inx] = 0

X_test_noise = np.zeros([k/bit_length, bit_length,2*coderate+1])
X_test_noise[:,:,0] = X_test[:,:,0] # True message bits
X_test_noise[:,bit_length-1,0] = np.zeros(X_test_noise[:,bit_length-1,0].shape) # Set the Last Bit to be 0. 
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
 
target    = X_test[:,0:bit_length-1,:].reshape([X_test.shape[0],X_test.shape[1]-1,1]) # Ignore the last bit (zero padding)

# BER
c_ber = 1- sum(sum(predicted == target))*\
       1.0/(target.shape[0] * target.shape[1] *target.shape[2])

# BLER
tp0 = (abs(np.round(predicted)-target)).reshape([target.shape[0],target.shape[1]]) 
bler = sum(np.sum(tp0,axis=1)>0)*1.0/(target.shape[0])

print 'test nn ber', c_ber[0]
print 'test nn bler', bler



# Save the BER and BLER
if print_errors == True:
    id = str(bit_length)+'_'+str(fsSNR)+'_'+str(nsSNR)    	
    np.savetxt('mv_'+id+str(np.random.random())+'.txt', [c_ber[0], bler], delimiter=',')

# Interpret
if run_us == True:
	execfile('us.py')
