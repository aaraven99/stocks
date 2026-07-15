import os,requests
from core.cache import DiskCache
MODELS=['mistralai/mistral-7b-instruct','meta-llama/llama-3.1-8b-instruct','google/gemma-7b-it','openchat/openchat-7b','nousresearch/hermes-2-pro-llama-3-8b']
def narrative(prompt):
 key=os.getenv('OPENROUTER_API_KEY')
 if not key:return {'text':'LLM disabled: deterministic quantitative evidence only.','model':None}
 cache=DiskCache('.cache/llm',ttl=86400);cache_key=__import__('hashlib').sha256(prompt.encode()).hexdigest();cached=cache.get(cache_key)
 if cached:return cached
 for model in MODELS:
  try:
   r=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0},timeout=20);r.raise_for_status();result={'text':r.json()['choices'][0]['message']['content'],'model':model};cache.set(cache_key,result);return result
  except Exception:continue
 return {'text':'LLM fallback chain unavailable.','model':None}
