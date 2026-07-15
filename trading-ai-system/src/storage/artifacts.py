from pathlib import Path
import json
from core.constants import ARTIFACTS
def ensure_artifacts(): ARTIFACTS.mkdir(parents=True,exist_ok=True); return ARTIFACTS
def write_json(name,payload):
 p=ensure_artifacts()/name; p.write_text(json.dumps(payload,indent=2,default=str)); return p
def write_text(name,text):
 p=ensure_artifacts()/name; p.write_text(text); return p
