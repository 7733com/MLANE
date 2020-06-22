#Hyperparameter Setting
The hyperparameter settings for baselines are described as follows:
\begin{itemize}

  \item DeepWalk. $embedding\  dimension = 128$. For node classification, $walk\  length = 80$, $window\  size = 10$, $walks\  per\  node = 40$. For link prediction and network reconstruction, $walk\  length = 40$, $walks\  per\  node = 10$, $window\  size = 5$. 

  \item node2vec. $embedding\  dimension = 128$. For node classification, $walk\  length = 80$, $window\  size = 10$, $walks\  per\  node = 40$. For link prediction and network reconstruction, $walk\  length = 40$, $walks\  per\  node = 10$, $window\  size = 5$. For Cora, $p = 0.25$, $q=1$. For Citeseer, $p = 4$, $q = 0.25$. For BlogCatalog, $p = 0.25$, $q=0.25$. For PPI, $p=4$, $q=1$. For Amazon, $p = 2$, $q = 1$. For HepTh, $p=4$, $q =1$.
  
  \item HOPE. $embedding\  dimension = 128$. We use RPR for constructing similarity matrices. For node classification, we set $\alpha = 0.5$. For link prediction and network reconstruction, we set $\alpha = 0.2$.

  \item SDNE. $embedding\  dimension = 128$. For node classification, $\alpha = 10^{-5}$, $\beta = 10$. For link prediction and network reconstruction, $\alpha = 10^{-5}$, $\beta = 5$.

  \item ProNE. $embedding\  dimension = 128$, $step=10$, $\theta=0.5$, $\mu=0.2$. 

  \item AttentionWalk. $embedding\  dimension = 128$, $\beta = 0.5$, $C = 5$.

  \item GAT. $dropout\  rate = 0.6$, $\lambda = 0.0005$, $K = 8$, $number\  of\  hidden\  units = 8$.

  \item LINE. $embedding\  dimension = 128$. The $number\  of\  negative\  samples = 5$.

  \item struc2vec. $walk\  length = 10$, $walks\  per\  node = 40$, $embedding\  dimension = 128$. 

  \item DRNE. $k = 16$, $\lambda = 0.5$, $S = 100$.

  \item RiWalk. $walk\  length = 80$, $window\  size = 10$, $walks\  per\  node = 10$, $neighborhood\  size = 4$, $embedding\  dimension = 128$. 

  \item Role2Vec. $walk\  length = 80$, $window\  size = 5$, $walks\  per\  node = 10$, $P = 1$, $Q = 1$, $embedding\  dimension = 128$. 

\end{itemize}
