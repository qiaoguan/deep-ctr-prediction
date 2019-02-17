# deep-ctr-prediction

一些广告算法(CTR预估)相关的DNN模型


wide&deep 可以参考tensorflow教程https://github.com/tensorflow/models/tree/master/official/wide_deep

deep&cross

deepfm

ESMM

Deep Interest Network

ResNet

代码使用tf.estimator构建, 数据存储为tfrecord格式(字典，key:value), 采用tf.Dataset API, 加快IO速度，支持工业级的应用。特征工程定义在input_fn,模型定义在model_fn,实现特征和模型代码分离,特征工程代码只用修改input_fn,模型代码只用修改model_fn。数据默认都是存在hadoop，可以根据自己需求存在本地

参考文献

【1】Heng-Tze Cheng, Levent Koc et all.   "Wide & Deep Learning for Recommender Systems,"   In 1st Workshop on Deep Learning for Recommender Systems,2016.

【2】Huifeng Guo et all.  "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," In IJCAI,2017.

【3】Ruoxi Wang et all.  "Deep & Cross Network for Ad Click Predictions,"  In ADKDD,2017.

【4】Xiao Ma et all.  "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate,"  In SIGIR,2018.

【5】Guorui Zhou et all.  "Deep Interest Network for Click-Through Rate Prediction," In KDD,2018.

【6】Kaiming He et all.  "Deep Residual Learning for Image Recognition," In CVPR,2016.