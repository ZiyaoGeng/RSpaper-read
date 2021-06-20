# RSpaper-read

`RSpaper-read`分享一些我读过的经典推荐论文，质量有保障，适合初学者入门。主要分为三部分内容：

1. Review：任何领域入门都少不了综述，推荐的文章包括了基于深度学习的推荐、图学习等；
2. Model：算法模型肯定是推荐领域的重点，按照不同阶段，再细化分为召回（matching）与排序（ranking）：
   - matching：召回阶段的模型面临的数据样本是整个物料库，所以它需要在低延时的前提下完成候选物品集的召回给排序阶段；
   - ranking：排序阶段区别于召回，要求模型更加复杂，重特征之间的交叉，主要的指标是CTR；
3. Others：推荐中其他的方向或者有趣的内容；



## [Review](./Abstract/review.md)

以下五篇综述都非常适合入门推荐系统：

|                            Paper                             |     Published in      | Time |
| :----------------------------------------------------------: | :-------------------: | :--: |
|   Deep Learning for Matching in Search and Recommendation    |         SIGIR         | 2018 |
| Deep Learning Based Recommender System: A Survey and New Perspectives | ACM Computing Surveys | 2019 |
|      Learning and Reasoning on Graph for Recommendation      |         CIKM          | 2019 |
|  Graph Learning Approaches to Recommender Systems: A Review  |         IJCAI         | 2021 |
| Sequential Recommender Systems: Challenges, Progress and Prospects |         AAAI          | 2019 |

&nbsp;

## Model

模型按照按照工业界来划分，召回和排序两个大块，由于粗排的文章没有读过，就先不加在里面了。

### [matching](./Abstract/matching.md)

召回阶段，工业界一般会采用多路召回的形式，即使是现在经常使用的基于向量化的召回，也只会作为其中的一路。多路召回的模型中，最常用的就是`ItemCF`（基于实际场景），现在工业界也经常会将其作为一路，毕竟又简单又好用。再往后，最经典的就是`Matrix Factorization`（矩阵分解），召回、排序都可以应用。

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
| Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model \| **SVD++** |     KDD      | 2008 |
| Matrix Factorization Techniques for Recommender Systems\|**MF** |     IEEE     | 2009 |
|   Neural network-based Collaborative Filtering \| **NCF**    |     WWW      | 2017 |

&nbsp;

再往后，就是基于向量化的召回模型（MF其实也算），**双塔模型**是其中最为通用的架构之一，下面三篇是具有浓厚工业风的文章，业界应用也非常多。

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
| Learning Deep Structured Semantic Models for Web Search using Clickthrough Data｜**DSSM** |     CIKM     | 2013 |
| Deep Neural Networks for YouTube Recommendations \|**YoutubeDNN** |    RecSys    | 2016 |
| Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations |    RecSys    | 2019 |

&nbsp;

目前，也有很多通过用户行为序列来表征用户，即**序列推荐**，这也是我个人的研究方向，文章包含了学术与工业。

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
| Factorizing personalized markov chains for next-basket recommendation \| **FMPC** |     KDD      | 2010 |
| Learning hierarchical representation model for nextbasket recommendation\|**HRM** |     IEEE     | 2015 |
| Translation-based recommendation: A scalable method for modeling sequential behavior \| **TransRec** |    IJCAI     | 2018 |
| Session-based Recommendation with Recurrent Neural Networks \| **GRU4Rec** |     ICLR     | 2016 |
| Recurrent neural networks with top-k gains for session-based recommendations \| **GRU4Rec+** |     WWW      | 2017 |
| Personalized top-n sequential recommendation via convolutional sequence embedding \| **Caser** |     ICDM     | 2018 |
|    Self-Attentive Sequential Recommendation \| **SASRec**    |     ICDM     | 2018 |
| STAMP: short-term attention/memory priority model for session-based recommendation｜**STAMP** |     KDD      | 2018 |
| Next item recommendation with self-attentive metric learning｜**AttRec** |     AAAI     | 2019 |
| BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer｜**BERT4Rec** |     CIKM     | 2019 |
| Multi-Interest Network with Dynamic Routing for Recommendation  at Tmall \| **MIND** |     CIKM     | 2019 |
| FISSA: fusing item similarity models with self-attention networks for sequential recommendation｜**FISSA** |    RecSys    | 2020 |
| SSE-PT: Sequential recommendation via personalized transforme｜**SSE-PT** |     KDD      | 2020 |
| Time Interval Aware Self-Attention for Sequential Recommendation｜**TiSASRec** |     WSDM     | 2020 |
| MEANTIME: Mixture of Attention Mechanisms with Multi-temporal Embeddings for Sequential Recommendation｜**MEANTIME** |    RecSys    | 2020 |
| Controllable Multi-Interest Framework for Recommendation \| **ComiRec** |     KDD      | 2020 |
|                          预训练模型                          |              |      |
| S3 -Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization｜**S3** |     CIKM     | 2020 |
| User BERT: self-supervised user representation learning｜**u-bert** |     ICLR     | 2021 |
|                            基于图                            |              |      |
| Session-Based Recommendation with Graph Neural Networks｜**SR-GNN** |     AAAI     | 2019 |

&nbsp;

### ranking

这里的ranking主要指的是精排部分的模型，

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
|               Factorization Machines \| **FM**               |     ICDM     | 2010 |
| Field-aware Factorization Machines for CTR Prediction｜**FFM** |    RecSys    | 2016 |
|    Wide & Deep Learning for Recommender Systems｜**WDL**     |     DLRS     | 2016 |
| Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features \| **Deep Crossing** |     KDD      | 2016 |
| Product-based Neural Networks for User Response Prediction \| **PNN** |     ICDM     | 2016 |
|   Deep & Cross Network for Ad Click Predictions \| **DCN**   |    ADKDD     | 2017 |
| Neural Factorization Machines for Sparse Predictive Analytics \| **NFM** |    SIGIR     | 2018 |
| Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks \| **AFM** |    IJCAI     | 2017 |
| DeepFM: A Factorization-Machine based Neural Network for CTR Prediction \| **DeepFM** |    IJCAI     | 2017 |
| xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems \| **xDeepFM** |     KDD      | 2018 |
| Deep Interest Network for Click-Through Rate Prediction \| **DIN** |     KDD      | 2018 |
| Outer Product-based Neural Collaboractive Filtering \| **ONN** |    IJCAI     | 2018 |
| Behavior Sequence Transformer for E-commerce Recommendation in Alibaba \| **BST** |   DLP-KDD    | 2019 |
| Deep Interest Evolution Network for Click-Through Rate Prediction ｜ **DIEN** |     AAAI     | 2019 |
| Deep Match to Rank Model for Personalized Click-Through Rate Prediction \| **DMR** |     AAAI     | 2020 |
|                           序列推荐                           |              |      |
| Sequential recommendation with user memory networks｜**MANN** |     WSDM     | 2018 |
|                            多任务                            |              |      |
| Entire Space Multi-Task Model: An Effective Approach for Estimation Post-Click Conversion Rate \| **ESMM** |    SIGIR     | 2018 |



## Others

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
| Neural Collaborative Filtering vs. Matrix Factorization Revisited |    RecSys    | 2020 |
|      FuxiCTR: An Open Benchmark for Click-Through Rate       |              | 2020 |

&nbsp;

## Contact Details

作者有一个自己的公众号：**推荐算法的小齿轮**，如果喜欢里面的内容，不妨点个关注。

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="30%"/></div>