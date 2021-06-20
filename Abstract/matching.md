# Matching Review



### [1] Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model



> Recommender systems provide users with personalized suggestions for products or services. These systems often rely on Collaborating Filtering (CF), where past transactions are analyzed in order to establish connections between users and products. The two more successful approaches to CF are latent factor models, which directly proﬁle both users and products, and neighborhood models, which analyze similarities between products or users. In this work we introduce some innovations to both approaches. The factor and neighborhood models can now be smoothly merged, thereby building a more accurate combined model. Further accuracy improvements are achieved by extending the models to exploit both explicit and implicit feedback by the users. The methods are tested on the Netﬂix data. Results are better than those previously published on that dataset. In addition, we suggest a new evaluation metric, which highlights the differences among methods, based on their performance at a top-K recommendation task.

SVD++



### [2] Matrix Factorization Techniques for Recommender Systems



> As the Netflix Prize competition has demonstrated, matrix factorization models are superior to classic nearest-neighbor techniques for producing product recommendations, allowing the incorporation of additional information such as implicit feedback, temporal effects, and confidence levels. 

非常经典的MF综述！



### [3]  Neural network-based Collaborative Filtering



> In recent years, deep neural networks have yielded immense success on speech recognition, computer vision and natural language processing. However, the exploration of deep neural networks on recommender systems has received relatively less scrutiny. In this work, we strive to develop techniques based on neural networks to tackle the key problem in recommendation — collaborative ﬁltering — on the basis of implicit feedback.
>
> Although some recent work has employed deep learning for recommendation, they primarily used it to model auxiliary information, such as textual descriptions of items and acoustic features of musics. When it comes to model the key factor in collaborative ﬁltering — the interaction between user and item features, they still resorted to matrix factorization and applied an inner product on the latent features of users and items.
>
> By replacing the inner product with a neural architecture that can learn an arbitrary function from data, we present a general framework named NCF, short for Neural networkbased Collaborative Filtering. NCF is generic and can express and generalize matrix factorization under its framework. To supercharge NCF modelling with non-linearities, we propose to leverage a multi-layer perceptron to learn the user–item interaction function. Extensive experiments on two real-world datasets show signiﬁcant improvements of our proposed NCF framework over the state-of-the-art methods. Empirical evidence shows that using deeper layers of neural networks oﬀers better recommendation performance.



### [4] Learning Deep Structured Semantic Models for Web Search using Clickthrough Data



> Latent semantic models, such as LSA, intend to map a query to its relevant documents at the semantic level where keyword-based matching often fails. In this study we strive to develop a series of new latent semantic models with a deep structure that project queries and documents into a common low-dimensional space where the relevance of a document given a query is readily computed as the distance between them. The proposed deep structured semantic models are discriminatively trained by maximizing the conditional likelihood of the clicked documents given a query using the clickthrough data. To make our models applicable to large-scale Web search applications, we also use a technique called word hashing, which is shown to effectively scale up our semantic models to handle large vocabularies which are common in such tasks. The new models are evaluated on a Web document ranking task using a real-world data set. Results show that our best model significantly outperforms other latent semantic models, which were considered state-of-the-art in the performance prior to the work presented in this paper.

虽然是搜索，但里面的架构在工业界应用很多



### [5] Deep Neural Networks for YouTube Recommendations



> YouTube represents one of the largest scale and most sophisticated industrial recommendation systems in existence. In this paper, we describe the system at a high level and focus on the dramatic performance improvements brought by deep learning. The paper is split according to the classic two-stage information retrieval dichotomy: ﬁrst, we detail a deep candidate generation model and then describe a separate deep ranking model. We also provide practical lessons and insights derived from designing, iterating and maintaining a massive recommendation system with enormous userfacing impact.

这篇太经典了，需要精读！



### [6] Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations



> Many recommendation systems retrieve and score items from a very large corpus. A common recipe to handle data sparsity and power-law item distribution is to learn item representations from its content features. Apart from many content-aware systems based on matrix factorization, we consider a modeling framework using two-tower neural net, with one of the towers (item tower) encoding a wide variety of item content features. A general recipe of training such two-tower models is to optimize loss functions calculated from in-batch negatives, which are items sampled from a random minibatch. However, in-batch loss is subject to sampling biases, potentially hurting model performance, particularly in the case of highly skewed distribution. In this paper, we present a novel algorithm for estimating item frequency from streaming data. Through theoretical analysis and simulation, we show that the proposed algorithm can work without requiring fixed item vocabulary, and is capable of producing unbiased estimation and being adaptive to item distribution change. We then apply the sampling-bias-corrected modeling approach to build a large scale neural retrieval system for YouTube recommendations. The system is deployed to retrieve personalized suggestions from a corpus with tens of millions of videos. We demonstrate the effectiveness of sampling-bias correction through offline experiments on two real-world datasets. We also conduct live A/B testings to show that the neural retrieval system leads to improved recommendation quality for YouTube.



### [7] Factorizing personalized markov chains for next-basket recommendation



> Recommender systems are an important component of many websites. Two of the most popular approaches are based on matrix factorization (MF) and Markov chains (MC). MF methods learn the general taste of a user by factorizing the matrix over observed user-item preferences. On the other hand, MC methods model sequential behavior by learning a transition graph over items that is used to predict the next action based on the recent actions of a user. In this paper, we present a method bringing both approaches together. Our method is based on personalized transition graphs over underlying Markov chains. That means for each user an own transition matrix is learned – thus in total the method uses a transition cube. As the observations for estimating the transitions are usually very limited, our method factorizes the transition cube with a pairwise interaction model which is a special case of the Tucker Decomposition. We show that our factorized personalized MC (FPMC) model subsumes both a common Markov chain and the normal matrix factorization model. For learning the model parameters, we introduce an adaption of the Bayesian Personalized Ranking (BPR) framework for sequential basket data. Empirically, we show that our FPMC model outperforms both the common matrix factorization and the unpersonalized MC model both learned with and without factorization.



### [8] Learning hierarchical representation model for nextbasket recommendation



> Next basket recommendation is a crucial task in market basket analysis. Given a user’s purchase history, usually a sequence of transaction data, one attempts to build a recommender that can predict the next few items that the user most probably would like. Ideally, a good recommender should be able to explore the sequential behavior (i.e., buying one item leads to buying another next), as well as account for users’ general taste (i.e., what items a user is typically interested in) for recommendation. Moreover, these two factors may interact with each other to inﬂuence users’ next purchase. To tackle the above problems, in this paper, we introduce a novel recommendation approach, namely hierarchical representation model (HRM). HRM can well capture both sequential behavior and users’ general taste by involving transaction and user representations in prediction. Meanwhile, the ﬂexibility of applying diﬀerent aggregation operations, especially nonlinear operations, on representations allows us to model complicated interactions among diﬀerent factors. Theoretically, we show that our model subsumes several existing methods when choosing proper aggregation operations. Empirically, we demonstrate that our model can consistently outperform the state-of-the-art baselines under diﬀerent evaluation metrics on real-world transaction data.



### [9] Translation-based recommendation: A scalable method for modeling sequential behavior



> Modeling the complex interactions between users and items is at the core of designing successful recommender systems. One key task consists of predicting users’ personalized sequential behavior, where the challenge mainly lies in modeling ‘third-order’ interactions between a user, her previously visited item(s), and the next item to consume. In this paper, we propose a uniﬁed method, TransRec, to model such interactions for largescale sequential prediction. Methodologically, we embed items into a ‘transition space’ where users are modeled as translation vectors operating on item sequences. Empirically, this approach outperforms the state-of-the-art on a wide spectrum of real-world datasets.



### [10] Session-based Recommendation with Recurrent Neural Networks



> We apply recurrent neural networks (RNN) on a new domain, namely recommender systems. Real-life recommender systems often face the problem of having to base recommendations only on short session-based data (e.g. a small sportsware website) instead of long user histories (as in the case of Netﬂix). In this situation the frequently praised matrix factorization approaches are not accurate. This problem is usually overcome in practice by resorting to item-to-item recommendations, i.e. recommending similar items. We argue that by modeling the whole session, more accurate recommendations can be provided. We therefore propose an RNNbased approach for session-based recommendations. Our approach also considers practical aspects of the task and introduces several modiﬁcations to classic RNNs such as a ranking loss function that make it more viable for this speciﬁc problem. Experimental results on two data-sets show marked improvements over widely used approaches.



### [11] Recurrent neural networks with top-k gains for session-based recommendations 



> RNNs have been shown to be excellent models for sequential data and in particular for data that is generated by users in an sessionbased manner. The use of RNNs provides impressive performance benefits over classical methods in session-based recommendations. In this work we introduce novel ranking loss functions tailored to RNNs in the recommendation setting. The improved performance of these losses over alternatives, along with further tricks and refinements described in this work, allow for an overall improvement of up to 35% in terms of MRR and Recall@20 over previous sessionbased RNN solutions and up to 53% over classical collaborative filtering approaches. Unlike data augmentation-based improvements, our method does not increase training times significantly. We further demonstrate the performance gain of the RNN over baselines in an online A/B test.



### [12] Personalized top-n sequential recommendation via convolutional sequence embedding



> Top-N sequential recommendation models each user as a sequence of items interacted in the past and aims to predict top-N ranked items that a user will likely interact in a “near future”. The order of interaction implies that sequential patterns play an important role where more recent items in a sequence have a larger impact on the next item. In this paper, we propose a Convolutional Sequence Embedding Recommendation Model (Caser) as a solution to address this requirement. The idea is to embed a sequence of recent items into an “image” in the time and latent spaces and learn sequential patterns as local features of the image using convolutional ﬁlters. This approach provides a uniﬁed and ﬂexible network structure for capturing both general preferences and sequential patterns. The experiments on public data sets demonstrated that Caser consistently outperforms state-of-the-art sequential recommendation methods on a variety of common evaluation metrics.



### [13] Self-Attentive Sequential Recommendation



> Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the ‘context’ of users’ activities on the basis of actions they have performed recently. To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). Markov Chains assume that a user’s next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. The goal of our work is to balance these two goals, by proposing a self-attention based sequential model (SASRec) that allows us to capture long-term semantics (like an RNN), but, using an attention mechanism, makes its predictions based on relatively few actions (like an MC). At each time step, SASRec seeks to identify which items are ‘relevant’ from a user’s action history, and use them to predict the next item. Extensive empirical studies show that our method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. Moreover, the model is an order of magnitude more efﬁcient than comparable CNN/RNN-based models. Visualizations on attention weights also show how our model adaptively handles datasets with various density, and uncovers meaningful patterns in activity sequences.



### [14] STAMP: short-term attention/memory priority model for session-based recommendation



> Predicting users’ actions based on anonymous sessions is a challenging problem in web-based behavioral modeling research, mainly due to the uncertainty of user behavior and the limited information. Recent advances in recurrent neural networks have led to promising approaches to solving this problem, with long short-term memory model proving effective in capturing users’ general interests from previous clicks. However, none of the existing approaches explicitly take the effects of users’ current actions on their next moves into account. In this study, we argue that a long-term memory model may be insufficient for modeling long sessions that usually contain user interests drift caused by unintended clicks. A novel short-term attention/memory priority model is proposed as a remedy, which is capable of capturing users’ general interests from the long-term memory of a session context, whilst taking into account users’ current interests from the short-term memory of the last-clicks. The validity and efficacy of the proposed attention mechanism is extensively evaluated on three benchmark data sets from the RecSys Challenge 2015 and CIKM Cup 2016. The numerical results show that our model achieves state-of-the-art performance in all the tests.



### [15] Next item recommendation with self-attentive metric learning



> In this paper, we propose a novel sequence-aware recommendation model. Our model utilizes self-aention mechanism to infer the item-item relationship from user’s historical interactions. With self-aention, it is able to estimate the relative weights of each item in user interaction trajectories to learn beer representations for user’s transient interests. e model is ﬁnally trained in a metric learning framework, taking both short-term and long-term intentions into consideration. Experiments on a wide range of datasets on diﬀerent domains demonstrate that our approach outperforms the state-of-the-art by a wide margin.



### [16] BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer



> Modeling users’ dynamic preferences from their historical behaviors is challenging and crucial for recommendation systems. Previous methods employ sequential neural networks to encode users’ historical interactions from left to right into hidden representations for making recommendations. Despite their effectiveness, we argue that such left-to-right unidirectional models are sub-optimal due to the limitations including: a) unidirectional architectures restrict the power of hidden representation in users’ behavior sequences; b) they often assume a rigidly ordered sequence which is not always practical. To address these limitations, we proposed a sequential recommendation model called BERT4Rec, which employs the deep bidirectional self-attention to model user behavior sequences. To avoid the information leakage and efficiently train the bidirectional model, we adopt the Cloze objective to sequential recommendation, predicting the random masked items in the sequence by jointly conditioning on their left and right context. In this way, we learn a bidirectional representation model to make recommendations by allowing each item in user historical behaviors to fuse information from both left and right sides. Extensive experiments on four benchmark datasets show that our model outperforms various state-of-the-art sequential models consistently.



### [17] Multi-Interest Network with Dynamic Routing for Recommendation  at Tmall



> Industrial recommender systems usually consist of the matching stage and the ranking stage, in order to handle the billion-scale of users and items. The matching stage retrieves candidate items relevant to user interests, while the ranking stage sorts candidate items by user interests. Thus, the most critical ability is to model and represent user interests for either stage. Most of the existing deep learning-based models represent one user as a single vector which is insufficient to capture the varying nature of user’s interests. In this paper, we approach this problem from a different view, to represent one user with multiple vectors encoding the different aspects of the user’s interests. We propose the Multi-Interest Network with Dynamic routing (MIND) for dealing with user’s diverse interests in the matching stage. Specifically, we design a multi-interest extractor layer based on capsule routing mechanism, which is applicable for clustering historical behaviors and extracting diverse interests. Furthermore, we develop a technique named label-aware attention to help learn a user representation with multiple vectors. Through extensive experiments on several public benchmarks and one largescale industrial dataset from Tmall, we demonstrate that MIND can achieve superior performance than state-of-the-art methods for recommendation. Currently, MIND has been deployed for handling major online traffic at the homepage on Mobile Tmall App.



### [18] FISSA: fusing item similarity models with self-attention networks for sequential recommendation



> Sequential recommendation has been a hot research topic because of its practicability and high accuracy by capturing the sequential information. As deep learning (DL) based methods being widely adopted to model the local and dynamic preferences beneath users’ behavior sequences, the modeling of users’ global and static preferences tends to be underestimated that usually, only some simple and crude users’ latent representations are introduced. Moreover, most existing methods hold an assumption that users’ intention can be fully captured by considering the historical behaviors, while neglect the possible uncertainty of users’ intention in reality, which may be influenced by the appearance of the candidate items to be recommended. In this paper, we thus focus on these two issues, i.e., the imperfect modeling of users’ global preferences in most DLbased sequential recommendation methods and the uncertainty of users’ intention brought by the candidate items, and propose a novel solution named fusing item similarity models with self-attention networks (FISSA) for sequential recommendation. Specifically, we treat the state-of-the-art self-attentive sequential recommendation (SASRec) model as the local representation learning module to capture the dynamic preferences beneath users’ behavior sequences in our FISSA, and further propose a global representation learning module to improve the modeling of users’ global preferences and a gating module that balances the local and global representations by taking the information of the candidate items into account. The global representation learning module can be seen as a locationbased attention layer, which is effective to fit in well with the parallelization training process of the self-attention framework. The gating module calculates the weight by modeling the relationship among the candidate item, the recently interacted item and the global preference of each user using an MLP layer. Extensive empirical studies on five commonly used datasets show that our FISSA significantly outperforms eight state-of-the-art baselines in terms of two commonly used metrics.



### [19] SSE-PT: Sequential recommendation via personalized transforme



> Temporal information is crucial for recommendation problems because user preferences are naturally dynamic in the real world. Recent advances in deep learning, especially the discovery of various attention mechanisms and newer architectures in addition to widely used RNN and CNN in natural language processing, have allowed for better use of the temporal ordering of items that each user has engaged with. In particular, the SASRec model, inspired by the popular Transformer model in natural languages processing, has achieved state-of-the-art results. However, SASRec, just like the original Transformer model, is inherently an un-personalized model and does not include personalized user embeddings. To overcome this limitation, we propose a Personalized Transformer (SSE-PT) model, outperforming SASRec by almost 5% in terms of NDCG@10 on 5 real-world datasets. Furthermore, after examining some random users’ engagement history, we find our model not only more interpretable but also able to focus on recent engagement patterns for each user. Moreover, our SSE-PT model with a slight modification, which we call SSE-PT++, can handle extremely long sequences and outperform SASRec in ranking results with comparable training speed, striking a balance between performance and speed requirements. Our novel application of the Stochastic Shared Embeddings (SSE) regularization is essential to the success of personalization. Code and data are open-sourced at https://github.com/wuliwei9278/SSE-PT.



### [20] Time Interval Aware Self-Attention for Sequential Recommendation



> Sequential recommender systems seek to exploit the order of users’ interactions, in order to predict their next action based on the context of what they have done recently. Traditionally, Markov Chains (MCs), and more recently Recurrent Neural Networks (RNNs) and Self Attention (SA) have proliferated due to their ability to capture the dynamics of sequential patterns. However a simplifying assumption made by most of these models is to regard interaction histories as ordered sequences, without regard for the time intervals between each interaction (i.e., they model the time-order but not the actual timestamp). In this paper, we seek to explicitly model the timestamps of interactions within a sequential modeling framework to explore the influence of different time intervals on next item prediction. We propose TiSASRec (Time Interval aware Self-attention based sequential recommendation), which models both the absolute positions of items as well as the time intervals between them in a sequence. Extensive empirical studies show the features of TiSASRec under different settings and compare the performance of self-attention with different positional encodings. Furthermore, experimental results show that our method outperforms various state-of-the-art sequential models on both sparse and dense datasets and different evaluation metrics.



### [21] MEANTIME: Mixture of Attention Mechanisms with Multi-temporal Embeddings for Sequential Recommendation



> Recently, self-attention based models have achieved state-of-the-art performance in sequential recommendation task. Following the custom from language processing, most of these models rely on a simple positional embedding to exploit the sequential nature of the user’s history. However, there are some limitations regarding the current approaches. First, sequential recommendation is different from language processing in that timestamp information is available. Previous models have not made good use of it to extract additional contextual information. Second, using a simple embedding scheme can lead to information bottleneck since the same embedding has to represent all possible contextual biases. Third, since previous models use the same positional embedding in each attention head, they can wastefully learn overlapping patterns. To address these limitations, we propose MEANTIME (MixturE of AtteNTIon mechanisms with Multi-temporal Embeddings) which employs multiple types of temporal embeddings designed to capture various patterns from the user’s behavior sequence, and an attention structure that fully leverages such diversity. Experiments on real-world data show that our proposed method outperforms current state-of-the-art sequential recommendation methods, and we provide an extensive ablation study to analyze how the model gains from the diverse positional information.



### [22] Controllable Multi-Interest Framework for Recommendation



> Recently, neural networks have been widely used in e-commerce recommender systems, owing to the rapid development of deep learning. We formalize the recommender system as a sequential recommendation problem, intending to predict the next items that the user might be interacted with. Recent works usually give an overall embedding from a user’s behavior sequence. However, a unified user embedding cannot reflect the user’s multiple interests during a period. In this paper, we propose a novel controllable multi-interest framework for the sequential recommendation, called ComiRec. Our multi-interest module captures multiple interests from user behavior sequences, which can be exploited for retrieving candidate items from the large-scale item pool. These items are then fed into an aggregation module to obtain the overall recommendation. The aggregation module leverages a controllable factor to balance the recommendation accuracy and diversity. We conduct experiments for the sequential recommendation on two real-world datasets, Amazon and Taobao. Experimental results demonstrate that our framework achieves significant improvements over state-of-the-art models 1 . Our framework has also been successfully deployed on the offline Alibaba distributed cloud platform.



### [23] S3 -Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization



> Recently, significant progress has been made in sequential recommendation with deep learning. Existing neural sequential recommendation models usually rely on the item prediction loss to learn model parameters or data representations. However, the model trained with this loss is prone to suffer from data sparsity problem. Since it overemphasizes the final performance, the association or fusion between context data and sequence data has not been well captured and utilized for sequential recommendation.
>
> To tackle this problem, we propose the model S 3 -Rec, which stands for Self-Supervised learning for Sequential Recommendation, based on the self-attentive neural architecture. The main idea of our approach is to utilize the intrinsic data correlation to derive self-supervision signals and enhance the data representations via pre-training methods for improving sequential recommendation. For our task, we devise four auxiliary self-supervised objectives to learn the correlations among attribute, item, subsequence, and sequence by utilizing the mutual information maximization (MIM) principle. MIM provides a unified way to characterize the correlation between different types of data, which is particularly suitable in our scenario. Extensive experiments conducted on six real-world datasets demonstrate the superiority of our proposed method over existing state-of-the-art methods, especially when only limited training data is available. Besides, we extend our self-supervised learning method to other recommendation models, which also improve their performance.



### [24] User BERT: self-supervised user representation learning



> This paper extends the BERT model to user data for pretraining user representations in a self-supervised way. By viewing actions (e.g., purchases and clicks) in behavior sequences (i.e., usage history) in an analogous way to words in sentences, we propose methods for the tokenization, the generation of input representation vectors and a novel pretext task to enable the pretraining model to learn from its own input, omitting the burden of collecting additional data. Further, our model adopts a uniﬁed structure to simultaneously learn from long-term and short-term user behavior as well as user proﬁles. Extensive experiments demonstrate that the learned representations result in signiﬁcant improvements when transferred to three different real-world tasks, particularly in comparison with task-speciﬁc modeling and representations obtained from multi-task learning.



### [25] Session-Based Recommendation with Graph Neural Networks



> The problem of session-based recommendation aims to predict user actions based on anonymous sessions. Previous methods model a session as a sequence and estimate user representations besides item representations to make recommendations. Though achieved promising results, they are insufficient to obtain accurate user vectors in sessions and neglect complex transitions of items. To obtain accurate item embedding and take complex transitions of items into account, we propose a novel method, i.e. Session-based Recommendation with Graph Neural Networks, SR-GNN for brevity. In the proposed method, session sequences are modeled as graphstructured data. Based on the session graph, GNN can capture complex transitions of items, which are difficult to be revealed by previous conventional sequential methods. Each session is then represented as the composition of the global preference and the current interest of that session using an attention network. Extensive experiments conducted on two real datasets show that SR-GNN evidently outperforms the state-of-the-art session-based recommendation methods consistently.





### [26] Sparse-Interest Network for Sequential Recommendation



> Recent methods in sequential recommendation focus on learning an overall embedding vector from a user’s behavior sequence for the next-item recommendation. However, from empirical analysis, we discovered that a user’s behavior sequence often contains multiple conceptually distinct items, while a unified embedding vector is primarily affected by one’s most recent frequent actions. Thus, it may fail to infer the next preferred item if conceptually similar items are not dominant in recent interactions. To this end, an alternative solution is to represent each user with multiple embedding vectors encoding different aspects of the user’s intentions. Nevertheless, recent work on multi-interest embedding usually considers a small number of concepts discovered via clustering, which may not be comparable to the large pool of item categories in real systems. It is a non-trivial task to effectively model a large number of diverse conceptual prototypes, as items are often not conceptually well clustered in fine granularity. Besides, an individual usually interacts with only a sparse set of concepts. In light of this, we propose a novel Sparse Interest NEtwork (SINE) for sequential recommendation. Our sparse-interest module can adaptively infer a sparse set of concepts for each user from the large concept pool and output multiple embeddings accordingly. Given multiple interest embeddings, we develop an interest aggregation module to actively predict the user’s current intention and then use it to explicitly model multiple interests for next-item prediction. Empirical results on several public benchmark datasets and one large-scale industrial dataset demonstrate that SINE can achieve substantial improvement over state-of-the-art methods.



### [27] SDM: Sequential Deep Matching Model for Online Large-scale Recommender System



> Capturing users’ precise preferences is a fundamental problem in large-scale recommender system. Currently, item-based Collaborative Filtering (CF) methods are common matching approaches in industry. However, they are not effective to model dynamic and evolving preferences of users. In this paper, we propose a new sequential deep matching (SDM) model to capture users’ dynamic preferences by combining short-term sessions and long-term behaviors. Compared with existing sequence-aware recommendation methods, we tackle the following two inherent problems in realworld applications: (1) there could exist multiple interest tendencies in one session. (2) long-term preferences may not be effectively fused with current session interests. Long-term behaviors are various and complex, hence those highly related to the short-term session should be kept for fusion. We propose to encode behavior sequences with two corresponding components: multi-head self-attention module to capture multiple types of interests and long-short term gated fusion module to incorporate long-term preferences. Successive items are recommended after matching between sequential user behavior vector and item embedding vectors. Offline experiments on real-world datasets show the superior performance of the proposed SDM. Moreover, SDM has been successfully deployed on online large-scale recommender system at Taobao and achieves improvements in terms of a range of commercial metrics.
