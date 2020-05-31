from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#PREPROCESSING FUNCTIONS

def tokenize(x):
    """
    Input is a List of sentences/strings to be tokenized
    Returns a Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tk = Tokenizer()
    tk.fit_on_texts(x)
    return tk.texts_to_sequences(x), tk


def pad(x, length=32):
    """
    Input is a List of sequences.
    Returns Padded numpy array of sequences
    """
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x):
    """
    Preprocess x:Feature List and y: Label List
    Applying tokenize() and pad()
    """
    preprocess_x, x_tk = tokenize(x)
    
    preprocess_x = pad(preprocess_x)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    #preprocess_y = y.reshape(*y.shape, 1)

    return preprocess_x, x_tk #, preprocess_y


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    logits: Logits from a neural network
    tokenizer: Keras Tokenizer fit on the labels
    Returns a String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])