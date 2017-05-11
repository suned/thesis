# semeval_baseline_position_dim_200
## Time	: 03-05-2017 12:23:39
### Auxiliary Tasks

### Hyper-Parameters

| Parameter              | Value |
|-----------------------:|:------|
| max-len                | 32  |
| trainable embedding    | False    |
| batch size             | 64    |
| patience               | 150    |
| dropout                | False    |
| filters                | 150    |
| n_grams                | [1, 2, 3, 4, 5]    |
| position embedding dim | 200    |

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
shared_word_embedding (Embedding (None, 32, 300)       402089100                                    
____________________________________________________________________________________________________
shared_position_embedding (Embed (None, 32, 200)       12800                                        
____________________________________________________________________________________________________
embedding_merge (Concatenate)    (None, 32, 700)       0                                            
____________________________________________________________________________________________________
shared_convolution_1_gram (Conv1 (None, 32, 150)       105150                                       
____________________________________________________________________________________________________
shared_convolution_2_gram (Conv1 (None, 31, 150)       210150                                       
____________________________________________________________________________________________________
shared_convolution_3_gram (Conv1 (None, 30, 150)       315150                                       
____________________________________________________________________________________________________
shared_convolution_4_gram (Conv1 (None, 29, 150)       420150                                       
____________________________________________________________________________________________________
shared_convolution_5_gram (Conv1 (None, 28, 150)       525150                                       
____________________________________________________________________________________________________
pooling_1_gram (GlobalMaxPooling (None, 150)           0                                            
____________________________________________________________________________________________________
pooling_2_gram (GlobalMaxPooling (None, 150)           0                                            
____________________________________________________________________________________________________
pooling_3_gram (GlobalMaxPooling (None, 150)           0                                            
____________________________________________________________________________________________________
pooling_4_gram (GlobalMaxPooling (None, 150)           0                                            
____________________________________________________________________________________________________
pooling_5_gram (GlobalMaxPooling (None, 150)           0                                            
____________________________________________________________________________________________________
convolution_merge (Concatenate)  (None, 750)           0                                            
____________________________________________________________________________________________________
SemEval_output (Dense)           (None, 19)            14269                                        
====================================================================================================
Total params: 403,691,919.0
Trainable params: 1,602,819.0
Non-trainable params: 402,089,100.0
____________________________________________________________________________________________________

```
### Validation Set Report
```
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)       0.86      0.74      0.79        65
      Cause-Effect(e2,e1)       0.88      0.83      0.85       133
   Component-Whole(e1,e2)       0.76      0.70      0.73        94
   Component-Whole(e2,e1)       0.59      0.66      0.62        92
 Content-Container(e1,e2)       0.76      0.93      0.84        70
 Content-Container(e2,e1)       0.94      0.83      0.88        35
Entity-Destination(e1,e2)       0.78      0.85      0.81       161
     Entity-Origin(e1,e2)       0.70      0.76      0.73       110
     Entity-Origin(e2,e1)       1.00      0.74      0.85        39
 Instrument-Agency(e1,e2)       1.00      0.25      0.40        24
 Instrument-Agency(e2,e1)       0.71      0.79      0.74        85
 Member-Collection(e1,e2)       0.69      0.64      0.67        14
 Member-Collection(e2,e1)       0.85      0.89      0.87       129
     Message-Topic(e1,e2)       0.82      0.79      0.81       107
     Message-Topic(e2,e1)       0.56      0.53      0.54        19
                    Other       0.46      0.47      0.46       285
  Product-Producer(e1,e2)       0.74      0.53      0.62        60
  Product-Producer(e2,e1)       0.62      0.66      0.64        77

              avg / total       0.72      0.71      0.71      1599

```