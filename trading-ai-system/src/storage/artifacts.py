from pathlib import Path
import json,os
from core.constants import ARTIFACTS
def artifact_root():
 configured=os.getenv('TRADING_ARTIFACTS_PATH')
 if configured:return Path(configured)
 return ARTIFACTS/'test_mode' if os.getenv('TEST_MODE','').lower() in ('1','true','yes') else ARTIFACTS
def ensure_artifacts():
 root=artifact_root();root.mkdir(parents=True,exist_ok=True);return root
def write_json(name,payload):
 p=ensure_artifacts()/name; p.write_text(json.dumps(payload,indent=2,default=str)); return p
def write_text(name,text):
 p=ensure_artifacts()/name; p.write_text(text); return p
