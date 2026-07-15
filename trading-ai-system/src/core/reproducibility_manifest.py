import platform,sys,subprocess
from datetime import datetime,timezone
from core.utils import stable_hash
def build_manifest(config,inputs): return {'run_id':stable_hash([config,inputs,datetime.now(timezone.utc).date().isoformat()])[:20],'created_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'config':config,'inputs':inputs,'seed':config.get('system',{}).get('seed',42)}
