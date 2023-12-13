# DFIL: Deepfake Incremental Learning by Exploiting Domain-invariant Forgery Clues
This is official code for DFIL.<br>
[Paper](https://arxiv.org/pdf/2309.09526.pdf)

# Overview of DFIL
![](https://github.com/DeepFakeIL/DFIL/blob/main/DFIL/img/overview.png)

# Detail of training
Firstly, you should download all datasets including(FF++,DFDC-p,CDF,DFD).<br>

Secondly, you can use file 'train_CNN_SupCon_and_CE.py' to train your first detection model with FF++ dataset.<br>

Thirdly, you can use 'get_feature.py' , 'get_image_info.py' and 'create_memory.py' to construct your memory set.<br>

Finally, you could randomly pick up 25 train video in your new dataset and add them into your memery set to train new model by file 'train_CNN_SupCon_and_CE.py'.<br>

# Trained Model
[First Task Model](https://drive.google.com/file/d/1tQls-XQsjDTEKMYrzC9mHJGAemQw9eOa/view?usp=drive_link)

# Citation
```
@inproceedings{pan2023dfil,
  title={DFIL: Deepfake Incremental Learning by Exploiting Domain-invariant Forgery Clues},
  author={Pan, Kun and Yin, Yifang and Wei, Yao and Lin, Feng and Ba, Zhongjie and Liu, Zhenguang and Wang, Zhibo and Cavallaro, Lorenzo and Ren, Kui},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8035--8046},
  year={2023}
}
```
