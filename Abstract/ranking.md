

## Ranking Review



### [1] Factorization Machines

> In this paper, we introduce Factorization Machines (FM) which are a new model class that combines the advantages of Support Vector Machines (SVM) with factorization models. Like SVMs, FMs are a general predictor working with any real valued feature vector. In contrast to SVMs, FMs model all interactions between variables using factorized parameters. Thus they are able to estimate interactions even in problems with huge sparsity (like recommender systems) where SVMs fail. We show that the model equation of FMs can be calculated in linear time and thus FMs can be optimized directly. So unlike nonlinear SVMs, a transformation in the dual form is not necessary and the model parameters can be estimated directly without the need of any support vector in the solution. We show the relationship to SVMs and the advantages of FMs for parameter estimation in sparse settings.
>
> On the other hand there are many different factorization models like matrix factorization, parallel factor analysis or specialized models like SVD++, PITF or FPMC. The drawback of these models is that they are not applicable for general prediction tasks but work only with special input data. Furthermore their model equations and optimization algorithms are derived individually for each task. We show that FMs can mimic these models just by specifying the input data (i.e. the feature vectors). This makes FMs easily applicable even for users without expert knowledge in factorization models.



### [2] Field-aware Factorization Machines for CTR Prediction



> Click-through rate (CTR) prediction plays an important role in computational advertising. Models based on degree-2 polynomial mappings and factorization machines (FMs) are widely used for this task. Recently, a variant of FMs, ﬁeldaware factorization machines (FFMs), outperforms existing models in some world-wide CTR-prediction competitions. Based on our experiences in winning two of them, in this paper we establish FFMs as an eﬀective method for classifying large sparse data including those from CTR prediction. First, we propose eﬃcient implementations for training FFMs. Then we comprehensively analyze FFMs and compare this approach with competing models. Experiments show that FFMs are very useful for certain classiﬁcation problems. Finally, we have released a package of FFMs for public use.



### [3] Wide & Deep Learning for Recommender Systems



> Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classiﬁcation problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are eﬀective and interpretable, while generalization requires more feature engineering eﬀort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the beneﬁts of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep signiﬁcantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.



### [4] Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features



> Manually crafted combinatorial features have been the “secret sauce” behind many successful models. For web-scale applications, however, the variety and volume of features make these manually crafted features expensive to create, maintain, and deploy. This paper proposes the Deep Crossing model which is a deep neural network that automatically combines features to produce superior models. The input of Deep Crossing is a set of individual features that can be either dense or sparse. The important crossing features are discovered implicitly by the networks, which are comprised of an embedding and stacking layer, as well as a cascade of Residual Units.
>
> Deep Crossing is implemented with a modeling tool called the Computational Network Tool Kit (CNTK), powered by a multi-GPU platform. It was able to build, from scratch, two web-scale models for a major paid search engine, and achieve superior results with only a sub-set of the features used in the production models. This demonstrates the potential of using Deep Crossing as a general modeling paradigm to improve existing products, as well as to speed up the development of new models with a fraction of the investment in feature engineering and acquisition of deep domain knowledge.



### [5] Product-based Neural Networks for User Response Prediction



> Predicting user responses, such as clicks and conversions, is of great importance and has found its usage in many Web applications including recommender systems, web search and online advertising. The data in those applications is mostly categorical and contains multiple ﬁelds; a typical representation is to transform it into a high-dimensional sparse binary feature representation via one-hot encoding. Facing with the extreme sparsity, traditional models may limit their capacity of mining shallow patterns from the data, i.e. low-order feature combinations. Deep models like deep neural networks, on the other hand, cannot be directly applied for the high-dimensional input because of the huge feature space. In this paper, we propose a Product-based Neural Networks (PNN) with an embedding layer to learn a distributed representation of the categorical data, a product layer to capture interactive patterns between interﬁeld categories, and further fully connected layers to explore high-order feature interactions. Our experimental results on two large-scale real-world ad click datasets demonstrate that PNNs consistently outperform the state-of-the-art models on various metrics.



### [6] Deep & Cross Network for Ad Click Predictions



> Feature engineering has been the key to the success of many prediction models. However, the process is nontrivial and oen requires manual feature engineering or exhaustive searching. DNNs are able to automatically learn feature interactions; however, they generate all the interactions implicitly, and are not necessarily eﬃcient in learning all types of cross features. In this paper, we propose the Deep & Cross Network (DCN) which keeps the beneﬁts of a DNN model, and beyond that, it introduces a novel cross network that is more eﬃcient in learning certain bounded-degree feature interactions. In particular, DCN explicitly applies feature crossing at each layer, requires no manual feature engineering, and adds negligible extra complexity to the DNN model. Our experimental results have demonstrated its superiority over the state-of-art algorithms on the CTR prediction dataset and dense classiﬁcation dataset, in terms of both model accuracy and memory usage.



### [7] Neural Factorization Machines for Sparse Predictive Analytics



> Many predictive tasks of web applications need to model categorical variables, such as user IDs and demographics like genders and occupations. To apply standard machine learning techniques, these categorical predictors are always converted to a set of binary features via one-hot encoding, making the resultant feature vector highly sparse. To learn from such sparse data eﬀectively, it is crucial to account for the interactions between features.
>
> Factorization Machines (FMs) are a popular solution for eﬃciently using the second-order feature interactions. However, FM models feature interactions in a linear way, which can be insuﬃcient for capturing the non-linear and complex inherent structure of real-world data. While deep neural networks have recently been applied to learn non-linear feature interactions in industry, such as the Wide&Deep by Google and DeepCross by Microso, the deep structure meanwhile makes them diﬃcult to train.
>
> In this paper, we propose a novel model Neural Factorization Machine (NFM) for prediction under sparse seings. NFM seamlessly combines the linearity of FM in modelling second-order feature interactions and the non-linearity of neural network in modelling higher-order feature interactions. Conceptually, NFM is more expressive than FM since FM can be seen as a special case of NFM without hidden layers. Empirical results on two regression tasks show that with one hidden layer only, NFM signiﬁcantly outperforms FM with a 7.3% relative improvement. Compared to the recent deep learning methods Wide&Deep and DeepCross, our NFM uses a shallower structure but oﬀers beer performance, being much easier to train and tune in practice.



### [8] Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks



> Factorization Machines (FMs) are a supervised learning approach that enhances the linear regression model by incorporating the second-order feature interactions. Despite effectiveness, FM can be hindered by its modelling of all feature interactions with the same weight, as not all feature interactions are equally useful and predictive. For example, the interactions with useless features may even introduce noises and adversely degrade the performance. In this work, we improve FM by discriminating the importance of different feature interactions. We propose a novel model named Attentional Factorization Machine (AFM), which learns the importance of each feature interaction from data via a neural attention network. Extensive experiments on two real-world datasets demonstrate the effectiveness of AFM. Empirically, it is shown on regression task AFM betters FM with a 8.6% relative improvement, and consistently outperforms the state-of-the-art deep learning methods Wide&Deep [ Cheng et al., 2016 ] and DeepCross [ Shan et al., 2016 ] with a much simpler structure and fewer model parameters. Our implementation of AFM is publicly available at: https://github. com/hexiangnan/attentional factorization machine



### [9] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction



> Learning sophisticated feature interactions behind user behaviors is critical in maximizing CTR for recommender systems. Despite great progress, existing methods seem to have a strong bias towards low- or high-order interactions, or require expertise feature engineering. In this paper, we show that it is possible to derive an end-to-end learning model that emphasizes both low- and highorder feature interactions. The proposed model, DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture. Compared to the latest Wide & Deep model from Google, DeepFM has a shared input to its “wide” and “deep” parts, with no need of feature engineering besides raw features. Comprehensive experiments are conducted to demonstrate the effectiveness and efﬁciency of DeepFM over the existing models for CTR prediction, on both benchmark data and commercial data.



### [10] xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems



> Combinatorial features are essential for the success of many commercial models. Manually crafting these features usually comes with high cost due to the variety, volume and velocity of raw data in web-scale systems. Factorization based models, which measure interactions in terms of vector product, can learn patterns of combinatorial features automatically and generalize to unseen features as well. With the great success of deep neural networks (DNNs) in various fields, recently researchers have proposed several DNNbased factorization model to learn both low- and high-order feature interactions. Despite the powerful ability of learning an arbitrary function from data, plain DNNs generate feature interactions implicitly and at the bit-wise level. In this paper, we propose a novel Compressed Interaction Network (CIN), which aims to generate feature interactions in an explicit fashion and at the vector-wise level. We show that the CIN share some functionalities with convolutional neural networks (CNNs) and recurrent neural networks (RNNs). We further combine a CIN and a classical DNN into one unified model, and named this new model eXtreme Deep Factorization Machine (xDeepFM). On one hand, the xDeepFM is able to learn certain bounded-degree feature interactions explicitly; on the other hand, it can learn arbitrary low- and high-order feature interactions implicitly. We conduct comprehensive experiments on three real-world datasets. Our results demonstrate that xDeepFM outperforms state-of-the-art models. We have released the source code of xDeepFM at https://github.com/Leavingseason/xDeepFM.



### [11] Deep Interest Network for Click-Through Rate Prediction



> Click-through rate prediction is an essential task in industrial applications, such as online advertising. Recently deep learning based models have been proposed, which follow a similar Embedding&MLP paradigm. In these methods large scale sparse input features are first mapped into low dimensional embedding vectors, and then transformed into fixed-length vectors in a group-wise manner, finally concatenated together to fed into a multilayer perceptron (MLP) to learn the nonlinear relations among features. In this way, user features are compressed into a fixed-length representation vector, in regardless of what candidate ads are. The use of fixed-length vector will be a bottleneck, which brings difficulty for Embedding&MLP methods to capture user’s diverse interests effectively from rich historical behaviors. In this paper, we propose a novel model: Deep Interest Network (DIN) which tackles this challenge by designing a local activation unit to adaptively learn the representation of user interests from historical behaviors with respect to a certain ad. This representation vector varies over different ads, improving the expressive ability of model greatly. Besides, we develop two techniques: mini-batch aware regularization and data adaptive activation function which can help training industrial deep networks with hundreds of millions of parameters. Experiments on two public datasets as well as an Alibaba real production dataset with over 2 billion samples demonstrate the effectiveness of proposed approaches, which achieve superior performance compared with state-of-the-art methods. DIN now has been successfully deployed in the online display advertising system in Alibaba, serving the main traffic.



### [12] Behavior Sequence Transformer for E-commerce Recommendation in Alibaba 



> Deep learning based methods have been widely used in industrial recommendation systems (RSs). Previous works adopt an Embedding&MLP paradigm: raw features are embedded into lowdimensional vectors, which are then fed on to MLP for final recommendations. However, most of these works just concatenate different features, ignoring the sequential nature of users’ behaviors. In this paper, we propose to use the powerful Transformer model to capture the sequential signals underlying users’ behavior sequences for recommendation in Alibaba. Experimental results demonstrate the superiority of the proposed model, which is then deployed online at Taobao and obtain significant improvements in online Click-Through-Rate (CTR) comparing to two baselines.



### [13] Deep Interest Evolution Network for Click-Through Rate Prediction



> Deep learning based methods have been widely used in industrial recommendation systems (RSs). Previous works adopt an Embedding&MLP paradigm: raw features are embedded into lowdimensional vectors, which are then fed on to MLP for final recommendations. However, most of these works just concatenate different features, ignoring the sequential nature of users’ behaviors. In this paper, we propose to use the powerful Transformer model to capture the sequential signals underlying users’ behavior sequences for recommendation in Alibaba. Experimental results demonstrate the superiority of the proposed model, which is then deployed online at Taobao and obtain significant improvements in online Click-Through-Rate (CTR) comparing to two baselines.



### [14] Deep Match to Rank Model for Personalized Click-Through Rate Prediction



> Click-through rate (CTR) prediction is a core task in the ﬁeld of recommender system and many other applications. For CTR prediction model, personalization is the key to improve the performance and enhance the user experience. Recently, several models are proposed to extract user interest from user behavior data which reﬂects user’s personalized preference implicitly. However, existing works in the ﬁeld of CTR prediction mainly focus on user representation and pay less attention on representing the relevance between user and item, which directly measures the intensity of user’s preference on target item. Motivated by this, we propose a novel model named Deep Match to Rank (DMR) which combines the thought of collaborative ﬁltering in matching methods for the ranking task in CTR prediction. In DMR, we design Userto-Item Network and Item-to-Item Network to represent the relevance in two forms. In User-to-Item Network, we represent the relevance between user and item by inner product of the corresponding representation in the embedding space. Meanwhile, an auxiliary match network is presented to supervise the training and push larger inner product to represent higher relevance. In Item-to-Item Network, we ﬁrst calculate the item-to-item similarities between user interacted items and target item by attention mechanism, and then sum up the similarities to obtain another form of user-to-item relevance. We conduct extensive experiments on both public and industrial datasets to validate the effectiveness of our model, which outperforms the state-of-art models signiﬁcantly.



### [15] Sequential recommendation with user memory networks



> User preferences are usually dynamic in real-world recommender systems, and a user’s historical behavior records may not be equally important when predicting his/her future interests. Existing recommendation algorithms – including both shallow and deep approaches – usually embed a user’s historical records into a single latent vector/representation, which may have lost the per item- or feature-level correlations between a user’s historical records and future interests. In this paper, we aim to express, store, and manipulate users’ historical records in a more explicit, dynamic, and effective manner. To do so, we introduce the memory mechanism to recommender systems. Specifically, we design a memory-augmented neural network (MANN) integrated with the insights of collaborative filtering for recommendation. By leveraging the external memory matrix in MANN, we store and update users’ historical records explicitly, which enhances the expressiveness of the model. We further adapt our framework to both item- and feature-level versions, and design the corresponding memory reading/writing operations according to the nature of personalized recommendation scenarios. Compared with state-of-the-art methods that consider users’ sequential behavior for recommendation, e.g., sequential recommenders with recurrent neural networks (RNN) or Markov chains, our method achieves significantly and consistently better performance on four real-world datasets. Moreover, experimental analyses show that our method is able to extract the intuitive patterns of how users’ future actions are affected by previous behaviors.



### [16] Entire Space Multi-Task Model: An Effective Approach for Estimation Post-Click Conversion Rate



> Estimating post-click conversion rate (CVR) accurately is crucial for ranking systems in industrial applications such as recommendation and advertising. Conventional CVR modeling applies popular deep learning methods and achieves state-of-the-art performance. However it encounters several task-specific problems in practice, making CVR modeling challenging. For example, conventional CVR models are trained with samples of clicked impressions while utilized to make inference on the entire space with samples of all impressions. This causes a sample selection bias problem. Besides, there exists an extreme data sparsity problem, making the model fitting rather difficult. In this paper, we model CVR in a brand-new perspective by making good use of sequential pattern of user actions, i.e., impression → click → conversion. The proposed Entire Space Multi-task Model (ESMM) can eliminate the two problems simultaneously by i) modeling CVR directly over the entire space, ii) employing a feature representation transfer learning strategy. Experiments on dataset gathered from traffic logs of Taobao’s recommender system demonstrate that ESMM significantly outperforms competitive methods. We also release a sampling version of this dataset to enable future research. To the best of our knowledge, this is the first public dataset which contains samples with sequential dependence of click and conversion labels for CVR modeling.



### [17] Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts



> Neural-based multi-task learning has been successfully used in many real-world large-scale applications such as recommendation systems. For example, in movie recommendations, beyond providing users movies which they tend to purchase and watch, the system might also optimize for users liking the movies afterwards. With multi-task learning, we aim to build a single model that learns these multiple goals and tasks simultaneously. However, the prediction quality of commonly used multi-task models is often sensitive to the relationships between tasks. It is therefore important to study the modeling tradeos between task-specic objectives and inter-task relationships.
>
> In this work, we propose a novel multi-task learning approach, Multi-gate Mixture-of-Experts (MMoE), which explicitly learns to model task relationships from data. We adapt the Mixture-ofExperts (MoE) structure to multi-task learning by sharing the expert submodels across all tasks, while also having a gating network trained to optimize each task. To validate our approach on data with dierent levels of task relatedness, we rst apply it to a synthetic dataset where we control the task relatedness. We show that the proposed approach performs better than baseline methods when the tasks are less related. We also show that the MMoE structure results in an additional trainability benet, depending on dierent levels of randomness in the training data and model initialization. Furthermore, we demonstrate the performance improvements by MMoE on real tasks including a binary classication benchmark, and a large-scale content recommendation system at Google.

