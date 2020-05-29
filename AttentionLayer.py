{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Layer\n",
    "import keras.backend as K\n",
    "\n",
    "class attention(Layer):\n",
    "    '''This class builds an attention layer.'''\n",
    "    \n",
    "    def __init__(self,**kwargs):\n",
    "        super(attention,self).__init__(**kwargs)\n",
    "\n",
    "    def build(self,input_shape):\n",
    "        '''\n",
    "        Define weights and biases. I.e.: If the previous LSTM layer’s output shape is (None, 32, 100)\n",
    "        then our output weight should be (100, 1) and bias should be (100, 1) dimensional.\n",
    "        '''\n",
    "        self.W=self.add_weight(name=\"att_weight\",shape=(input_shape[-1],1),initializer=\"normal\")\n",
    "        self.b=self.add_weight(name=\"att_bias\",shape=(input_shape[1],1),initializer=\"zeros\")        \n",
    "        super(attention, self).build(input_shape)\n",
    "\n",
    "    def call(self,x):\n",
    "        '''\n",
    "         Multi-Layer Perceptron. Takes the dot product of weights and inputs followed by the\n",
    "         addition of bias terms. Apply ‘tanh’ followed by softmax layer (alignment scores).\n",
    "         It's dimension will be the number of hidden states in the LSTM.\n",
    "         Taking its dot product along with the hidden states will provide the context vector.\n",
    "        '''\n",
    "        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)\n",
    "        at=K.softmax(et)\n",
    "        at=K.expand_dims(at,axis=-1)\n",
    "        output=x*at\n",
    "        return K.sum(output,axis=1)\n",
    "\n",
    "    def compute_output_shape(self,input_shape):\n",
    "        return (input_shape[0],input_shape[-1])\n",
    "\n",
    "    def get_config(self):\n",
    "        return super(attention,self).get_config()\n",
    "    \n",
    "    \n",
    "# Sources:\n",
    "# https://arxiv.org/abs/1706.03762\n",
    "# https://arxiv.org/abs/1601.06733\n",
    "# https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
