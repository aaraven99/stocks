import numpy as np
def select(features,max_features=30):
 keys=[k for k,v in features.items() if np.isfinite(v)]; return keys[:max_features]
