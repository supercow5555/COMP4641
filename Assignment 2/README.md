## Preface: 
This assignment was amazing. I went through highs and lows of effortfully learning these libraries in as detailed as possible and right on the cusp of the limits that I can comprehend.

## Edge Betweenness Contributions for Paths Starting from a Node

This repository contains the implementation of a  graph analysis  task based on  edge betweenness contributions  for paths starting from Node C. The program was completed to further my knowledge, so I can learn how to first code a simple undirected graph and then automate it to show the Girvan-Newman Method for community detection in networks.
### Overview

The Girvan-Newman method is a divisive, top-down hierarchical clustering algorithm that iteratively removes edges with **high betweenness centrality** to partition the network into communities. The **betweenness centrality** of an edge is the number of **shortest paths** between pairs of nodes that pass through that edge. By removing edges with high betweenness centrality, the algorithm breaks up the network into smaller clusters.

The graph used for the analysis task is a simple  undirected graph  with 7 nodes and 9 edges. The goal of the task was to calculate the edge  betweenness  contributions for paths starting from Node C.

### Procedure

The following procedure was followed to complete the task:

1.  Draw the  BFS  result for paths starting from Node C.
2.  Obtain the number of  shortest paths  from Node C to each node and show this number next to each node.
3.  Obtain the betweenness contributions for each path and show this number next to each edge.

The results of the analysis were put together into a single diagram.

### Progress

We have started the implementation of the Girvan-Newman method using  Python  and NetworkX library. We have created a  Jupyter notebook  `GirvanNewman.ipynb`  to demonstrate the progress so far. The notebook includes the following:

1.  Loading the graph and displaying the below.
2.  Computing the betweenness centrality of edges

**Todo:** 

3.  Removing the edge with the highest betweenness centrality
4.  Repeating steps 2-3 until the desired number of clusters is obtained

### Results

The diagram containing the results of the analysis is shown below.

The diagram shows the  BFS result  for paths starting from Node C, with the number of shortest paths from Node C to each node shown next to each node. The  betweenness contributions  for each path are also shown next to each edge.

### Conclusion

The edge betweenness contributions for paths starting from Node C were successfully calculated and will be soon  displayed in a single diagram. ***The process followed in this analysis can be extended to larger and more complex graphs to gain insights into the relationships and patterns within the data.***


### References

The lecture notes on graph analysis and edge betweenness contributions were used as a reference for completing this task.

## *GraphSAGE* on Zachary Karate Club Dataset
This repository contains the implementation of GraphSAGE on the **Zachary Karate Club** dataset using the **Deep Graph Library** (DGL). The goal is to learn **node embeddings** and visualize them using Matplotlib. The project also includes k-means clustering to partition the graph into subgraphs.

### Dataset
**Zachary Karate Club Dataset** is used in this project.

### Model
The **GraphSAGE model** is implemented using dgl.nn.SAGEConv from DGL. The model uses the **mean aggregator** and **ReLU activation function** for the first layer. The implementation is done for two scenarios:

 1. Two **GNN layers** with adjacency-based similarity function in the
    loss.
 2. Three GNN layers with 2-hop similarity function in the loss.

### Steps
1.  Build the GraphSAGE model
 - Two GNN layers are built using the provided update scheme and ReLU as
   the
 - **activation function** for the first layer.
The adjacency-based similarity function is used in the loss for training node embeddings.

2. Visualize node embeddings
Node embeddings are visualized using **Matplotlib**, similar to the output on page 3 (right) of **nrltutorial-part1.pdf.**

3. K-means clustering
The k-means clustering algorithm sklearn.cluster.KMeans is used to partition the graph into two subgraphs.
The nodes in the **embedding space** are shown in red and blue.

4. Repeat steps with 3 GNN layers and 2-hop similarity function
The above steps are repeated with 3 GNN layers and a 2-hop similarity function (replacing the adjacency-based similarity function in step 1 with 2-hop neighbors) in the loss, as described on page 18 of nrltutorial-part2.pdf.

### Conclusion

In this project, we implemented GraphSAGE on the  Zachary  karate club  dataset and evaluated two models with different configurations. We found that the 3-layer GraphSAGE model with 2-hop similarity function in the loss performs better in partitioning the graph into two subgraphs than the 2-layer model with adjacency-based similarity function. 

**3-layer GraphSAGE model with 2-hop similarity function:** 
In epoch 99, loss: 0.0846032202243805

 **2-layer model with adjacency-based similarity function**
 In epoch 99, loss: 0.11986055970191956

The results suggest that  deeper models  with more expressive power can capture more complex patterns in the data and improve the performance of  community detection.

### Dependencies
Deep Graph Library (DGL)
PyTorch
Matplotlib
scikit-learn

### References

-   "Inductive Representation Learning on  Large Graphs" by William L. Hamilton, Rex Ying, and Jure Leskovec. NIPS 2017.
-   DGL documentation:  [https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.SAGEConv.html](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.SAGEConv.html)
-   sklearn.cluster.KMeans documentation:  [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
-   Zachary  karate club dataset:  [http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm](http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm)


## Node Classification on Cora Dataset

### Overview

This model performs **node classification** on the Cora citation network dataset. The task is to predict the  ground truth category  (one of 7 categories) of each node (paper) based on the citation network.

### Model

The model used is a **2-layer  GraphSAGE** model  followed by a **linear classifier.**  GraphSAGE  is a  GNN  that learns node embeddings  by sampling and aggregating features from neighbor nodes. The loss function is the cross-entropy loss.

The general (so far partial) pipeline is:

1.  Load Cora dataset using DGL library and preprocess it to obtain the graph and node features.
2.  Define 2-layer GraphSAGE model with  ReLU activation  for first layer.
3.  Add a  linear layer  for classification.
4.  Train model  using **cross-entropy loss** and **Adam optimizer**.
5. Split the data into train, validation, and test sets.
6.  Evaluate model accuracy on value and test set.

### Conclusion

**Output:**
In epoch 99, loss: 0.000, val acc: 0.738, test acc: 0.742

The model is able to achieve 74.2% accuracy on the test set and 73.8% on validation set, indicating it learns meaningful  node representations  for classification based on the citation network structure. The results demonstrate the effectiveness of GraphSAGE in learning node representations for the task of node classification.

### References

- McCallum, A., Nigam, K., et al. (2000). Automating the construction of internet portals with machine learning. Information Retrieval, 3(2), 127-163.
-   Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In Advances in neural information processing systems (pp. 1024-1034).
-   GraphSAGE:  [https://arxiv.org/abs/1706.02216](https://arxiv.org/abs/1706.02216)
-   DGL library:  [https://www.dgl.ai/](https://www.dgl.ai/)

-   Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., & Eliassi-Rad, T. (2008, August).  Collective classification  in network data. In AI magazine (Vol. 29, No. 3, pp. 93-93).
-   DGL documentation:  [https://docs.dgl.ai/](https://docs.dgl.ai/)
-   Cora dataset:  [https://linqs.soe.ucsc.edu/data](https://linqs.soe.ucsc.edu/data)
-   PyTorch documentation:  [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

### Dependencies
-   Python 3.6+
-   PyTorch
-   DGL
-   NumPy

### Acknowledgements

The partial code for the implementation was provided as part of a university course assignment. The Cora dataset was sourced from DGL's built-in dataset module. 

Special thanks to *Srijan Saxena* who helped me immensly on the on Zachary Karate Club and Cora Dataset.


## Final remarks
The fact that I can put in the effort to learn this and understand it in detail and succeed shows me that I can do anything And that feeling is, simply put, fucking amazing. 
