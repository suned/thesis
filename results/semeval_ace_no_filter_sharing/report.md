# semeval_ace_no_filter_sharing
## Time	: 04-05-2017 17:37:23
### Auxiliary Tasks
ACE
### Hyper-Parameters

| Parameter              | Value |
|-----------------------:|:------|
| batch size             | 64    |
| patience               | 20    |
| dropout                | False    |
| filters                | 150    |
| n_grams                | [1, 2, 3, 4, 5]    |
| position embedding dim | 50    |

### SemEval Model Summary
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
word_input (InputLayer)          (None, 31)            0                                            
____________________________________________________________________________________________________
position1_input (InputLayer)     (None, 31)            0                                            
____________________________________________________________________________________________________
position2_input (InputLayer)     (None, 31)            0                                            
____________________________________________________________________________________________________
shared_word_embedding (Embedding multiple              410134500                                    
____________________________________________________________________________________________________
shared_position_embedding (Embed multiple              23900                                        
____________________________________________________________________________________________________
embeddings (Concatenate)         (None, 31, 400)       0                                            
____________________________________________________________________________________________________
convolution_1_gram (Conv1D)      (None, 31, 150)       60150                                        
____________________________________________________________________________________________________
convolution_2_gram (Conv1D)      (None, 30, 150)       120150                                       
____________________________________________________________________________________________________
convolution_3_gram (Conv1D)      (None, 29, 150)       180150                                       
____________________________________________________________________________________________________
convolution_4_gram (Conv1D)      (None, 28, 150)       240150                                       
____________________________________________________________________________________________________
convolution_5_gram (Conv1D)      (None, 27, 150)       300150                                       
____________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalMa (None, 150)           0                                            
____________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalMa (None, 150)           0                                            
____________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalMa (None, 150)           0                                            
____________________________________________________________________________________________________
global_max_pooling1d_4 (GlobalMa (None, 150)           0                                            
____________________________________________________________________________________________________
global_max_pooling1d_5 (GlobalMa (None, 150)           0                                            
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 750)           0                                            
____________________________________________________________________________________________________
SemEval_output (Dense)           (None, 19)            14269                                        
====================================================================================================
Total params: 411,073,419.0
Trainable params: 411,073,419.0
Non-trainable params: 0.0
____________________________________________________________________________________________________

```
### Validation Set Report
```
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)       0.88      0.78      0.83        65
      Cause-Effect(e2,e1)       0.86      0.85      0.86       133
   Component-Whole(e1,e2)       0.80      0.72      0.76        94
   Component-Whole(e2,e1)       0.62      0.60      0.61        92
 Content-Container(e1,e2)       0.85      0.81      0.83        70
 Content-Container(e2,e1)       0.88      0.86      0.87        35
Entity-Destination(e1,e2)       0.74      0.90      0.81       161
     Entity-Origin(e1,e2)       0.73      0.77      0.75       110
     Entity-Origin(e2,e1)       0.97      0.79      0.87        39
 Instrument-Agency(e1,e2)       1.00      0.46      0.63        24
 Instrument-Agency(e2,e1)       0.73      0.80      0.76        85
 Member-Collection(e1,e2)       0.75      0.64      0.69        14
 Member-Collection(e2,e1)       0.85      0.91      0.88       129
     Message-Topic(e1,e2)       0.85      0.82      0.83       107
     Message-Topic(e2,e1)       0.62      0.42      0.50        19
                    Other       0.52      0.50      0.51       285
  Product-Producer(e1,e2)       0.69      0.73      0.71        60
  Product-Producer(e2,e1)       0.65      0.68      0.66        77

              avg / total       0.74      0.73      0.73      1599

```