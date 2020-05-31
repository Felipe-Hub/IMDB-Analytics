{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.text import Tokenizer\n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "\n",
    "\n",
    "#PREPROCESSING FUNCTIONS\n",
    "\n",
    "def tokenize(x):\n",
    "    \"\"\"\n",
    "    Input is a List of sentences/strings to be tokenized\n",
    "    Returns a Tuple of (tokenized x data, tokenizer used to tokenize x)\n",
    "    \"\"\"\n",
    "    tk = Tokenizer()\n",
    "    tk.fit_on_texts(x)\n",
    "    return tk.texts_to_sequences(x), tk\n",
    "\n",
    "\n",
    "def pad(x, length):\n",
    "    \"\"\"\n",
    "    Input is a List of sequences.\n",
    "    Returns Padded numpy array of sequences\n",
    "    \"\"\"\n",
    "    return pad_sequences(x, maxlen=length, padding='post')\n",
    "\n",
    "\n",
    "def preprocess(x, y):\n",
    "    \"\"\"\n",
    "    Preprocess x:Feature List and y: Label List\n",
    "    Applying tokenize() and pad()\n",
    "    \"\"\"\n",
    "    preprocess_x, x_tk = tokenize(x)\n",
    "    preprocess_y, y_tk = tokenize(y)\n",
    "\n",
    "    preprocess_x = pad(preprocess_x)\n",
    "    preprocess_y = pad(preprocess_y)\n",
    "\n",
    "    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions\n",
    "    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)\n",
    "\n",
    "    return preprocess_x, preprocess_y, x_tk, y_tk\n",
    "\n",
    "\n",
    "def logits_to_text(logits, tokenizer):\n",
    "    \"\"\"\n",
    "    Turn logits from a neural network into text using the tokenizer\n",
    "    logits: Logits from a neural network\n",
    "    tokenizer: Keras Tokenizer fit on the labels\n",
    "    Returns a String that represents the text of the logits\n",
    "    \"\"\"\n",
    "    index_to_words = {id: word for word, id in tokenizer.word_index.items()}\n",
    "    index_to_words[0] = '<PAD>'\n",
    "\n",
    "    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])"
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
 "nbformat_minor": 2
}
