# DFIL: Deepfake Incremental Learning by Exploiting Domain-invariant Forgery Clues
This is official code for DFIL.<br>
[Paper](https://arxiv.org/pdf/2309.09526.pdf)

# Overview of DFIL
![](https://github.com/DeepFakeIL/DFIL/blob/main/DFIL/img/overview.png)

# Detail of train
Firstly, you should download all datasets including(FF++,DFDC-p,CDF,DFD).<br>

Secondly, you can use file 'train_CNN_SupCon_and_CE.py' to train your first detection model with FF++ dataset.<br>

Then, you can use 'get_feature.py' , 'get_image_info.py' and 'create_memory.py' to construct your memory set.<br>

Finaly, you could randomly pick up 25 train video in your new dataset and add them into your memery set to train new model by file 'train_CNN_SupCon_and_CE.py'.<br>

