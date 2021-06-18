# RSpaper-read

`RSpaper-read`分享一些我读过的经典推荐论文，质量有保障，适合初学者入门。主要分为三部分内容：

1. Review：任何领域入门都少不了综述，推荐的文章包括了基于深度学习的推荐、图学习等；
2. Model：算法模型肯定是推荐领域的重点，按照不同阶段，再细化分为召回（matching）与排序（ranking）：
   - matching：召回阶段的模型面临的数据样本是整个物料库，所以它需要在低延时的前提下完成候选物品集的召回给排序阶段；
   - ranking：排序阶段区别于召回，要求模型更加复杂，重特征之间的交叉，主要的指标是CTR；
3. Others：推荐中其他的方向或者有趣的内容；



## [Review](./Abstractreview.md)

|                            Paper                             |     Published in      | Author |
| :----------------------------------------------------------: | :-------------------: | :----: |
|   Deep Learning for Matching in Search and Recommendation    |         SIGIR         |  2018  |
| Deep Learning Based Recommender System: A Survey and New Perspectives | ACM Computing Surveys |  2019  |
|      Learning and Reasoning on Graph for Recommendation      |         CIKM          |  2019  |
|  Graph Learning Approaches to Recommender Systems: A Review  |         IJCAI         |  2021  |

&nbsp;

## Model

### General

|                            Paper                             | Published in | Time |   Category    |
| :----------------------------------------------------------: | :----------: | :--: | :-----------: |
| Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model \| **SVD++** |     KDD      | 2008 |    General    |
| Matrix Factorization Techniques for Recommender Systems\|**MF** |     IEEE     | 2009 |    General    |
| Deep Neural Networks for YouTube Recommendations \| **DNN Youtube** |     ACM      | 2016 | Deep Learning |
|   Neural network-based Collaborative Filtering \| **NCF**    |     WWW      | 2017 |    DL、CF     |

&nbsp;

### Sequence System

|                            Paper                             | Published in | Time |      Category      |
| :----------------------------------------------------------: | :----------: | :--: | :----------------: |
| Factorizing personalized markov chains for next-basket recommendation \| **FMPC** |     KDD      | 2010 |      General       |
| Learning hierarchical representation model for nextbasket recommendation\|**HRM** |     IEEE     | 2015 |      General       |
| Translation-based recommendation: A scalable method for modeling sequential behavior \| **TransRec** |    IJCAI     | 2018 |   Deep Learning    |
| Session-based Recommendation with Recurrent Neural Networks \| **GRU4Rec** |     ICLR     | 2016 |        RNN         |
| Recurrent neural networks with top-k gains for session-based recommendations \| **GRU4Rec+** |     WWW      | 2017 |       DL、CF       |
| Personalized top-n sequential recommendation via convolutional sequence embedding \| **Caser** |     ICDM     | 2018 |   Deep Learning    |
| Session-based recommendation with graph neural networks｜**SR-GNN** |     AAAI     | 2019 |        GNN         |
|    Self-Attentive Sequential Recommendation \| **SASRec**    |     ICDM     | 2018 |   Deep Learning    |
| STAMP: short-term attention/memory priority model for session-based recommendation｜**STAMP** |     KDD      | 2018 |     Attention      |
| Next item recommendation with self-attentive metric learning｜**AttRec** |     AAAI     | 2019 |     Attention      |
| BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer｜**BERT4Rec** |     CIKM     | 2019 |     Attention      |
| FISSA: fusing item similarity models with self-attention networks for sequential recommendation｜**FISSA** |    RecSys    | 2020 |     Attention      |
| SSE-PT: Sequential recommendation via personalized transforme｜**SSE-PT** |     KDD      | 2020 |     Attention      |
| Time Interval Aware Self-Attention for Sequential Recommendation｜**TiSASRec** |     WSDM     | 2020 |      Att+time      |
| Déjà vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation｜**CTA** |     WWW      | 2020 |      Att+time      |
| MEANTIME: Mixture of Attention Mechanisms with Multi-temporal Embeddings for Sequential Recommendation｜**MEANTIME** |    RecSys    | 2020 |      Att+time      |
| Multi-Interest Network with Dynamic Routing for Recommendation  at Tmall \| **MIND** |     CIKM     | 2019 | DL、Multi Interest |
| Controllable Multi-Interest Framework for Recommendation \| **ComiRec** |     KDD      | 2020 | DL、Multi Interest |
|  Sequential recommendation with user memory networks｜MANN   |     WSDM     | 2018 |         MN         |
| Towards neural mixture recommender for long range dependent user sequences\|**M3** |     WWW      | 2019 |      Mixture       |

&nbsp;

### CTR

|                            Paper                             | Published in | Time |     Category      |
| :----------------------------------------------------------: | :----------: | :--: | :---------------: |
|               Factorization Machines \| **FM**               |     ICDM     | 2010 |      General      |
| Field-aware Factorization Machines for CTR Prediction｜**FFM** |    RecSys    | 2016 |      General      |
|    Wide & Deep Learning for Recommender Systems｜**WDL**     |     DLRS     | 2016 |        DL         |
| Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features \| **Deep Crossing** |     KDD      | 2016 |        DL         |
| Product-based Neural Networks for User Response Prediction \| **PNN** |     ICDM     | 2016 |        DL         |
|   Deep & Cross Network for Ad Click Predictions \| **DCN**   |    ADKDD     | 2017 |        DL         |
| Neural Factorization Machines for Sparse Predictive Analytics \| **NFM** |    SIGIR     | 2018 |        DL         |
| Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks \| **AFM** |    IJCAI     | 2017 |        DL         |
| DeepFM: A Factorization-Machine based Neural Network for CTR Prediction \| **DeepFM** |    IJCAI     | 2017 |        DL         |
| xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems \| **xDeepFM** |     KDD      | 2018 |        DL         |
| Deep Interest Network for Click-Through Rate Prediction \| **DIN** |     KDD      | 2018 |        DL         |
| Outer Product-based Neural Collaboractive Filtering \| **ONN** |    IJCAI     | 2018 |        DL         |
| Behavior Sequence Transformer for E-commerce Recommendation in Alibaba \| **BST** |   DLP-KDD    | 2019 |  DL、Transformer  |
| Deep Interest Evolution Network for Click-Through Rate Prediction ｜ **DIEN** |     AAAI     | 2019 | DL、RNN、Interest |
| Deep Match to Rank Model for Personalized Click-Through Rate Prediction \| **DMR** |     AAAI     | 2020 |   Deep Learning   |

&nbsp;

### Multi-Task

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
| Entire Space Multi-Task Model: An Effective Approach for Estimation Post-Click Conversion Rate \| **ESMM** |    SIGIR     | 2018 |

&nbsp;

## Others

|                            Paper                             | Published in | Time |
| :----------------------------------------------------------: | :----------: | :--: |
| Neural Collaborative Filtering vs. Matrix Factorization Revisited |    RecSys    | 2020 |
|      FuxiCTR: An Open Benchmark for Click-Through Rate       |              | 2020 |

&nbsp;

## Contact Details

作者有一个自己的公众号：**推荐算法的小齿轮**，如果喜欢里面的内容，不妨点个关注。

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="30%"/></div>