"""Construct an actual finite mixture, not a variance-destroying weighted average."""
import numpy as np


def combine(samples,weights):
 keys=list(samples);n=min(len(samples[key]) for key in keys);counts={key:int(np.floor(max(weights.get(key,0),0)*n)) for key in keys};remainder=n-sum(counts.values())
 for key in sorted(keys,key=lambda item:weights.get(item,0),reverse=True)[:remainder]:counts[key]+=1
 pieces=[]
 for offset,key in enumerate(keys):
  count=counts[key]
  if count:pieces.append(np.asarray(samples[key])[(np.arange(count)*len(keys)+offset)%len(samples[key])])
 mixture=np.concatenate(pieces) if pieces else np.zeros(n);return mixture[np.argsort((np.arange(len(mixture))*2654435761)%max(len(mixture),1))]
