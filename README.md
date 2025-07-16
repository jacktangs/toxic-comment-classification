# 有害数据分类
本项目源于NTU EE6405NLP课程设计。主要实现的功能有：

利用LSTM，BERT做二分类，将评论分为有害（toxic）和无害（non-toxic）两种。支持单条评论分类和用户自己上传CSV数据集。若数据集同时包含标签，则可以得到准确度和混淆矩阵等指标，便于比较该任务中训练的LSTM和BERT两模型的性能。

利用LSTM和T5做多标签分类，根据置信度将评论打上有害（toxic），严重有害（severe-toxic），淫秽（obscene），恐吓（threat），侮辱（insult）和身份仇恨（identity_hate）的标签，多标签可以共存。

使用的数据集包括
1. ) [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) 来自Kaggle比赛的数据集，由维基百科评论页的有害评论组成。
2. ) [Sensai](https://www.kaggle.com/api/v1/datasets/download/uetchy/sensai) 由虚拟主播直播中的有害评论组成。
3. ) [The Toxicity Dataset](https://github.com/surge-ai/toxicity) 由来自各种流行的社交媒体平台的500条有害和500条无害评论组成。

上传了数据处理，模型训练和ui部分的代码。

实际效果如下图：
<img width="1280" height="649" alt="二分类" src="https://github.com/user-attachments/assets/63c798e6-044c-415d-be11-0fe8ae58f189" />

<img width="1278" height="649" alt="多标签分类" src="https://github.com/user-attachments/assets/45e9f2af-d989-4471-9733-de33abf76563" />
