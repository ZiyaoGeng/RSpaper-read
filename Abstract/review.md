# Review Abstract



## [1] Deep Learning for Matching in Search and Recommendation



> Matching is the key problem in both search and recommendation, that is to measure the relevance of a document to a query or the interest of a user on an item. Previously, machine learning methods have been exploited to address the problem, which learns a matching function from labeled data, also referred to as “learning to match” [21]. In recent years, deep learning has been successfully applied to matching and significant progresses have been made. Deep semantic matching models for search [25] and neural collaborative filtering models for recommendation [12] are becoming the state-of-the-art technologies. The key to the success of the deep learning approach is its strong ability in learning of representations and generalization of matching patterns from raw data (e.g., queries, documents, users, and items, particularly in their raw forms). In this tutorial, we aim to give a comprehensive survey on recent progress in deep learning for matching in search and recommendation. Our tutorial is unique in that we try to give a unified view on search and recommendation. In this way, we expect researchers from the two fields can get deep understanding and accurate insight on the spaces, stimulate more ideas and discussions, and promote developments of technologies.
>
> The tutorial mainly consists of three parts. Firstly, we introduce the general problem of matching, which is fundamental in both search and recommendation. Secondly, we explain how traditional machine learning techniques are utilized to address the matching problem in search and recommendation. Lastly, we elaborate how deep learning can be effectively used to solve the matching problems in both tasks.



来自何向南老师的一篇综述，包含了搜索与推荐两个部分的内容，当然题目的Match并不是推荐中“召回”的意思，而是指匹配。主体的内容以ppt的形式呈现，结构非常清晰，阅读完有很大的收获。



## [2] Deep Learning Based Recommender System: A Survey and New Perspectives



> With the growing volume of online information, recommender systems have been an effective strategy to overcome information overload. The utility of recommender systems cannot be overstated, given their widespread adoption in many web applications, along with their potential impact to ameliorate many problems related to over-choice. In recent years, deep learning has garnered considerable interest in many research fields such as computer vision and natural language processing, owing not only to stellar performance but also to the attractive property of learning feature representations from scratch. The influence of deep learning is also pervasive, recently demonstrating its effectiveness when applied to information retrieval and recommender systems research. The field of deep learning in recommender system is flourishing. This article aims to provide a comprehensive review of recent research efforts on deep learning-based recommender systems. More concretely, we provide and devise a taxonomy of deep learning-based recommendation models, along with a comprehensive summary of the state of the art. Finally, we expand on current trends and provide new perspectives pertaining to this new and exciting development of the field.



打开我基于深度学习的推荐系统的大门就是这篇综述，提及了一些较为常见的算法模型，适合入门。



## [3] Learning and Reasoning on Graph for Recommendation



> Recommendation methods construct predictive models to estimate the likelihood of a user-item interaction. Previous models largely follow a general supervised learning paradigm — treating each interaction as a separate data instance and performing prediction based on the “information isolated island”. Such methods, however, overlook the relations among data instances, which may result in suboptimal performance especially for sparse scenarios. Moreover, the models built on a separate data instance only can hardly exhibit the reasons behind a recommendation, making the recommendation process opaque to understand.
>
> In this tutorial, we revisit the recommendation problem from the perspective of graph learning. Common data sources for recommendation can be organized into graphs, such as user-item interactions (bipartite graphs), social networks, item knowledge graphs (heterogeneous graphs), among others. Such a graph-based organization connects the isolated data instances, bringing benefits to exploiting high-order connectivities that encode meaningful patterns for collaborative filtering, content-based filtering, social influence modeling and knowledge-aware reasoning. Together with the recent success of graph neural networks (GNNs), graph-based models have exhibited the potential to be the technologies for next-generation recommendation systems. This tutorial provides a review on graph-based learning methods for recommendation, with special focus on recent developments of GNNs and knowledge graph-enhanced recommendation. By introducing this emerging and promising topic in this tutorial, we expect the audience to get deep understanding and accurate insight on the spaces, stimulate more ideas and discussions, and promote developments of technologies.



依旧是何向南教授的文章，对比[1]，多了图神经网络的内容。



## [4] Graph Learning Approaches to Recommender Systems: A Review



> Recent years have witnessed the fast development of the emerging topic of Graph Learning based Recommender Systems (GLRS). GLRS mainly employ the advanced graph learning approaches to model users’ preferences and intentions as well as items’ characteristics and popularity for Recommender Systems (RS). Differently from conventional RS, includeng content based filtering and collaborative filtering, GLRS are built on simple or complex graphs where various objects, e.g., users, items, and attributes, are explicitly or implicitly connected. With the rapid development of graph learning, exploring and exploiting homogeneous or heterogeneous relations in graphs is a promising direction for building advanced RS. In this paper, we provide a systematic review of GLRS, on how they obtain the knowledge from graphs to improve the accuracy, reliability and explainability for recommendations. First, we characterize and formalize GLRS, and then summarize and categorize the key challenges in this new research area. Then, we survey the most recent and important developments in the area. Finally, we share some new research directions in this vibrant area.



是一篇比较短的图学习综述，可以简单看看入门。



## [5] Sequential Recommender Systems: Challenges, Progress and Prospects



> The emerging topic of sequential recommender systems (SRSs) has attracted increasing attention in recent years. Different from the conventional recommender systems (RSs) including collaborative filtering and content-based filtering, SRSs try to understand and model the sequential user behaviors, the interactions between users and items, and the evolution of users’ preferences and item popularity over time. SRSs involve the above aspects for more precise characterization of user contexts, intent and goals, and item consumption trend, leading to more accurate, customized and dynamic recommendations. In this paper, we provide a systematic review on SRSs. We first present the characteristics of SRSs, and then summarize and categorize the key challenges in this research area, followed by the corresponding research progress consisting of the most recent and representative developments on this topic. Finally, we discuss the important research directions in this vibrant area.



少见的序列推荐综述。

