import numpy as np
def ensemble_uncertainty(probabilities): return float(np.std(probabilities))
def confidence(probability,uncertainty): return float(max(0,min(1,1-uncertainty*3))*abs(probability-.5)*2)
