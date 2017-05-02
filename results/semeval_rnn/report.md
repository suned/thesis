# semeval_rnn
## Time	: 01-05-2017 15:46:02
### Auxiliary Tasks

### Hyper-Parameters

| Parameter              | Value |
|-----------------------:|:------|
| max-len                | 32  |
| trainable embedding    | True    |
| batch size             | 64    |
| patience               | 100    |
| dropout                | False    |
| filters                | 150    |
| n_grams                | [1, 2, 3, 4, 5]    |
| position embedding dim | 50    |

### SemEval Model Summary
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
word_input (InputLayer)          (None, 32)            0                                            
____________________________________________________________________________________________________
position1_input (InputLayer)     (None, 32)            0                                            
____________________________________________________________________________________________________
position2_input (InputLayer)     (None, 32)            0                                            
____________________________________________________________________________________________________
shared_word_embedding (Embedding (None, 32, 300)       408628800                                    
____________________________________________________________________________________________________
position_embedding (Embedding)   (None, 32, 50)        3200                                         
____________________________________________________________________________________________________
embedding_layer (Concatenate)    (None, 32, 400)       0                                            
____________________________________________________________________________________________________
bi_lstm (Bidirectional)          (None, 32, 300)       661200                                       
____________________________________________________________________________________________________
lstm (LSTM)                      (None, 150)           270600                                       
____________________________________________________________________________________________________
SemEval_output (Dense)           (None, 19)            2869                                         
====================================================================================================
Total params: 409,566,669.0
Trainable params: 409,566,669.0
Non-trainable params: 0.0
____________________________________________________________________________________________________

```
### Validation Set Report
```
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)       0.81      0.68      0.74        65
      Cause-Effect(e2,e1)       0.76      0.80      0.78       133
   Component-Whole(e1,e2)       0.57      0.57      0.57        94
   Component-Whole(e2,e1)       0.47      0.33      0.38        92
 Content-Container(e1,e2)       0.67      0.83      0.74        70
 Content-Container(e2,e1)       0.71      0.57      0.63        35
Entity-Destination(e1,e2)       0.78      0.83      0.81       161
     Entity-Origin(e1,e2)       0.76      0.59      0.67       110
     Entity-Origin(e2,e1)       1.00      0.08      0.14        39
 Instrument-Agency(e1,e2)       0.50      0.04      0.08        24
 Instrument-Agency(e2,e1)       0.36      0.42      0.39        85
 Member-Collection(e1,e2)       0.00      0.00      0.00        14
 Member-Collection(e2,e1)       0.74      0.86      0.80       129
     Message-Topic(e1,e2)       0.71      0.60      0.65       107
     Message-Topic(e2,e1)       0.67      0.11      0.18        19
                    Other       0.34      0.42      0.37       285
  Product-Producer(e1,e2)       0.49      0.37      0.42        60
  Product-Producer(e2,e1)       0.28      0.45      0.35        77

              avg / total       0.59      0.57      0.56      1599

```
