# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.


## Problem Statement and Dataset

We aim to develop an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text. The dataset used has a number of sentences, and each words have their tags. We have to vectorize these words using Embedding techniques to train our model.Bidirectional Recurrent Neural Networks connect two hidden layers of opposite directions to the same output.


![DL 6D](https://github.com/Rama-Lekshmi/named-entity-recognition/assets/118541549/3865cf1b-5690-4d1e-aff8-d91ef7e0a820)

### STEP 1:
Download and load the dataset to colab.
### STEP 2:
Scale the data using MinMaxScaler
### STEP 3:
Split the data into train and test.
### STEP 4:
Build the LSTM based recurrent neural networkv
### STEP 5:
Train the model with training data
### STEP 6:
Evaluate the model with the testing data
### STEP 7:
Plot the Stock prediction plot

## PROGRAM

~~~py
Developed by: RAMA E.K. LEKSHMI
Reg No:212222240082
~~~

### Importing Libraries

```py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
```

### Reading the dataset

```py
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
```

### Listing the words and tags

```py
words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
print("Unique tags are:", tags)
num_words = len(words)
num_tags = len(tags)
```

### Function to grouping the sentences

```py
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences
```

### Enumeration

```py
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
plt.hist([len(s) for s in sentences], bins=50)
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
max_len = 50
```

### Reshaping

```py
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
```

### Model Creation

```py
input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words,
                                   output_dim=50,
                                   input_length=max_len)(input_word)
dropout_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
bidirectional_lstm = layers.Bidirectional(
    layers.LSTM(units=100, return_sequences=True,
                recurrent_dropout=0.1))(dropout_layer)
output = layers.TimeDistributed(
    layers.Dense(num_tags, activation="softmax"))(bidirectional_lstm)                                                
model = Model(input_word, output)  
```

### Summary,Compiling and fitting

```py
model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32, 
    epochs=3,
)
print("RAMA E.K. LEKSHMI")
print("212222240082")
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
```

### Prediction Sequence

```py
print("RAMA E.K. LEKSHMI")
print("212222240082")
i = 79
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![DL 6 A](https://github.com/Rama-Lekshmi/named-entity-recognition/assets/118541549/f05a1175-d73c-4c35-90e0-914fe135d6df)

![DL 6B](https://github.com/Rama-Lekshmi/named-entity-recognition/assets/118541549/967424b1-98f4-41e4-a825-56afa730b696)

### Sample Text Prediction

![DL 6C](https://github.com/Rama-Lekshmi/named-entity-recognition/assets/118541549/aa31471d-d13a-4d8c-8af3-abdf27e11ef4)

## RESULT

Thus an LSTM-based model for recognizing the named entities in the text is successfully developed.
