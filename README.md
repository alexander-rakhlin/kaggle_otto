# Kaggle Otto Group Product Classification Challenge

My solution that scored 0.42232 and finished [Otto competition](https://www.kaggle.com/c/otto-group-product-classification-challenge) on 218th position out of 3514 teams.  
Represents 1:1 blend of XGBoost model and average of 20 Neural Nets. Models hyper parameters, NN architecture and blend weights have been chosen manually.

Requires:
* [XGBoost](https://github.com/dmlc/xgboost)
* [Theano](http://deeplearning.net/software/theano/)
* [Keras](https://github.com/fchollet/keras), [forum link](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13632/achieve-0-48-in-5-min-with-a-deep-net-feat-batchnorm-prelu/74125#post74125)

# Other work
Other Kagglers insights I found particularly interesting. For the most part they relate to blending. I list them here for further study:

1. **Triskelion**. Competition 62nd. Blending  
[forum link 1](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14297/share-your-models/79286#post79286)  
[forum link 2](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14297/share-your-models/79425#post79425)  
[Ensemble Selection from Libraries of Models](other_work/caruana.icml04.icdm06long.pdf)  
For his turn he is referring to another kaggler Emanuele Olivetti, [(forked code)](https://github.com/emanuele/kaggle_pbr)

2. **Hoang Duong**. Competition 6th. Blending  
[forum link](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14296/competition-write-up-optimistically-convergent/79384#post79384)  
[documentation](other_work/summary.pdf)

3. **Adam Harasimowicz**. Competition 66th. Blending, Hyperopt  
[forked code](https://github.com/alexander-rakhlin/kaggle_otto-adam-)  
[Blog post](http://blog.aicry.com/kaggle-otto-group-product-classification-challenge/)

4. **Mike Kim**. Competition 8th. T-SNE features and meta bagging  
[forum link](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14295/41599-via-tsne-meta-bagging/79080#post79080)  
[code](other_work/ottoHomeBagG4.R)
