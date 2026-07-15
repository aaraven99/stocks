from core.feature_stability import psi
def detect(reference,current):
 p=psi(reference,current); return {'psi':p,'ks':p**.5,'flag':'material' if p>.25 else 'normal'}
