from keras.layers import Layer
import keras.backend as K

class Attention(Layer):
    '''This class builds an attention layer.'''
    
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        '''
        Define weights and biases. I.e.: If the previous LSTM layer’s output shape is (None, 32, 100)
        then our output weight should be (100, 1) and bias should be (100, 1) dimensional.
        '''
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(Attention, self).build(input_shape)

    def call(self,x):
        '''
         Multi-Layer Perceptron. Takes the dot product of weights and inputs followed by the
         addition of bias terms. Apply ‘tanh’ followed by softmax layer (alignment scores).
         It's dimension will be the number of hidden states in the LSTM.
         Taking its dot product along with the hidden states will provide the context vector.
        '''
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()
    
    
# Sources:
# https://arxiv.org/abs/1706.03762
# https://arxiv.org/abs/1601.06733
# https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/