# deep-ctr-prediction

一些广告算法(CTR预估)相关的DNN模型


* wide&deep 可以参考official/wide_deep

* deep&cross

* deepfm

* ESMM

* Deep Interest Network

* ResNet

* xDeepFM

* AFM(Attentional FM)

* Transformer

* FiBiNET

代码使用tf.estimator构建, 数据存储为tfrecord格式(字典，key:value), 采用tf.Dataset API, 加快IO速度，支持工业级的应用。特征工程定义在input_fn,模型定义在model_fn,实现特征和模型代码分离,特征工程代码只用修改input_fn,模型代码只用修改model_fn。数据默认都是存在hadoop，可以根据自己需求存在本地, 特征工程和数据的处理可以参考Google开源的wide&deep模型(不使用tfrecord格式, 代码在official/wide_deep)

All codes are written based on tf.estimator API, the data is stored in tfrecord format(dictionary, key:value), and the tf.Dataset API is used to speed up IO speed, it support industrial applications. Feature engineering is defined in input_fn, model function is defined in model_fn, the related code of feature engineering and model function is completely separated,
the data is stored in hadoop by default, and can be locally stored according to your 
own need. The feature engineering and data processing can refer to Google's open source wide&deep model(without tfrecord format, codes are available at official/wide_deep)

# Requirements
* Tensorflow 1.10

# 参考文献

【1】Heng-Tze Cheng, Levent Koc et all.   "Wide & Deep Learning for Recommender Systems,"   In 1st Workshop on Deep Learning for Recommender Systems,2016.

【2】Huifeng Guo et all.  "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," In IJCAI,2017.

【3】Ruoxi Wang et all.  "Deep & Cross Network for Ad Click Predictions,"  In ADKDD,2017.

【4】Xiao Ma et all.  "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate,"  In SIGIR,2018.

【5】Guorui Zhou et all.  "Deep Interest Network for Click-Through Rate Prediction," In KDD,2018.

【6】Kaiming He et all.  "Deep Residual Learning for Image Recognition," In CVPR,2016.

【7】Jianxun Lian et all.  "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems,"  In KDD,2018.

【8】Jun Xiao et all. "Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks," In IJCAI, 2017.

【9】Ashish Vasmani et all.  "Attention is All You Need,"  In NIPS, 2017.

【10】Tongwen et all. "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction," In RecSys, 2019.