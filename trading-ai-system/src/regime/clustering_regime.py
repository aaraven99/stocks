import numpy as np
from sklearn.cluster import KMeans
def cluster_proxy(returns):
 """Unsupervised clustering of rolling return/volatility states, deterministic by seed."""
 r=np.asarray(returns.dropna(),dtype=float)
 if len(r)<60:return {'cluster':1,'trend':float(np.mean(r)) if len(r) else 0.,'volatility':float(np.std(r)) if len(r) else 0.,'method':'insufficient_history'}
 windows=np.array([[r[max(0,i-20):i].mean(),r[max(0,i-20):i].std()] for i in range(20,len(r)+1)])
 model=KMeans(n_clusters=min(4,len(windows)),n_init=20,random_state=42).fit(windows);state=int(model.labels_[-1]);center=model.cluster_centers_[state]
 return {'cluster':state,'trend':float(center[0]),'volatility':float(center[1]),'centers':model.cluster_centers_.tolist(),'method':'kmeans_rolling_return_volatility'}
