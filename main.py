import tensorflow as tf 
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

sent = [
    "This is the first sentence", 
    "Rivindu learns Machine Learning", 
    "Rivindu masters Data Science", 
    "I'm a good boy", 
    "Krish's videos are good"
]

voc_size = 500

onehot_repr = [one_hot(sentence, voc_size) for sentence in sent]

sent_length = 8
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

dimensions = 10
model = Sequential()

model.add(Embedding(input_dim=voc_size, 
                    output_dim=dimensions, 
                    input_length=sent_length))

model.compile(optimizer='adam', loss='mae')

model.build(input_shape=(None, sent_length))
model.summary()

print(model(embedded_docs))