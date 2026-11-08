"""Hydrate PG1, load it offline on H100, and run one synthetic forward anchor."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .s3_cache_hydration import run as hydrate_cache, validate_spec as validate_hydration


class ModelSmokeError(RuntimeError):
    pass


def _hydration_view(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        **spec,
        "scope": "s3_worker_cache_hydration_only",
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "model_loads": 0,
        "gpu_work": 0,
    }


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    validate_hydration(_hydration_view(spec))
    if spec.get("scope") != "pg1_model_forward_smoke_only":
        raise ModelSmokeError("model smoke scope changed")
    if spec.get("uses_gpu") is not True or spec.get("gpu_work_class") != "single_model_forward_smoke":
        raise ModelSmokeError("model smoke GPU boundary changed")
    if (
        spec.get("scientific_claim") is not False
        or int(spec.get("synthetic_items", -1)) != 1
        or int(spec.get("discovery_items_read", -1)) != 0
        or int(spec.get("confirmation_items_read", -1)) != 0
        or int(spec.get("science_items_evaluated", -1)) != 0
        or int(spec.get("model_loads", -1)) != 1
        or int(spec.get("gpu_work", -1)) != 1
    ):
        raise ModelSmokeError("model smoke must remain one synthetic non-science forward")
    if spec.get("forward_contract") != {
        "attention": "eager", "use_cache": False, "dtype": "bfloat16", "offline": True
    }:
        raise ModelSmokeError("model smoke forward contract changed")
    return spec


REMOTE_SMOKE = r'''import json,math,os,sys,time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['HF_DATASETS_OFFLINE']='1'
import torch
from PIL import Image
from transformers import AutoProcessor,PaliGemmaForConditionalGeneration
root=sys.argv[1]; started=time.monotonic()
assert torch.cuda.is_available() and torch.cuda.device_count()==1
processor=AutoProcessor.from_pretrained(root,local_files_only=True)
model=PaliGemmaForConditionalGeneration.from_pretrained(root,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
image=Image.new('RGB',(224,224),(17,29,43)); prompt='<image>caption en'; suffix='a dark square'
batch=processor(images=[image],text=[prompt],suffix=[suffix],padding=True,return_tensors='pt')
required={'input_ids','attention_mask','pixel_values','labels','token_type_ids'}; assert required.issubset(batch)
batch={k:v.to('cuda') if hasattr(v,'to') else v for k,v in batch.items()}
with torch.inference_mode(): output=model(**batch,use_cache=False,return_dict=True)
mask=batch['labels'][:,1:].ne(-100); selected=output.logits[:,:-1,:][mask]
assert selected.numel()>0 and bool(torch.isfinite(selected).all()) and bool(torch.isfinite(output.loss))
components=model.model
report={'status':'GREEN_PG1_MODEL_FORWARD_SMOKE','elapsed_seconds':time.monotonic()-started,'model_type':model.config.model_type,'architectures':model.config.architectures,'dtype':str(next(model.parameters()).dtype),'device':str(next(model.parameters()).device),'cuda_device_name':torch.cuda.get_device_name(0),'parameter_count':sum(p.numel() for p in model.parameters()),'decoder_layers':len(components.language_model.layers),'vision_tower_class':type(components.vision_tower).__name__,'projector_class':type(components.multi_modal_projector).__name__,'language_model_class':type(components.language_model).__name__,'input_shape':list(batch['input_ids'].shape),'pixel_shape':list(batch['pixel_values'].shape),'logit_shape':list(output.logits.shape),'suffix_tokens':int(mask.sum()),'loss':float(output.loss),'logit_abs_max':float(selected.float().abs().max()),'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),'attention':'eager','use_cache':False,'offline':True,'synthetic_anchor':{'image_rgb':[17,29,43],'image_size':[224,224],'prompt':'caption en','suffix':'a dark square'},'model_loads':1,'synthetic_items':1,'discovery_items_read':0,'confirmation_items_read':0,'science_items_evaluated':0,'gpu_work':True}
print(json.dumps(report,sort_keys=True))'''


def run(spec: dict[str, Any], *, url_file: Path, worker_host: str, worker_port: int,
        worker_key: Path, worker_result_root: str) -> dict[str, Any]:
    validate_spec(spec)
    hydration = hydrate_cache(
        _hydration_view(spec), url_file=url_file,
        worker_host=worker_host, worker_port=worker_port, worker_key=worker_key,
        worker_result_root=worker_result_root.rstrip("/") + "/hydration",
    )
    ssh = ["ssh", "-i", str(worker_key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=10", "-p", str(worker_port), f"root@{worker_host}"]
    command = "python3 -c " + shlex.quote(REMOTE_SMOKE) + " " + shlex.quote(spec["worker_snapshot_root"])
    completed = subprocess.run(ssh + [command], capture_output=True, timeout=int(spec["model_timeout_seconds"]))
    if completed.returncode != 0:
        raise ModelSmokeError("offline PG1 model smoke failed: " + completed.stderr.decode(errors="replace")[-1000:])
    smoke = json.loads(completed.stdout.decode())
    if smoke.get("status") != "GREEN_PG1_MODEL_FORWARD_SMOKE":
        raise ModelSmokeError("PG1 model smoke result is not GREEN")
    result = {"schema_version": 1, "run_id": spec["run_id"], "status": "GREEN",
              "hydration": hydration, "model_smoke": smoke, "automatic_retry": False}
    root = worker_result_root.rstrip("/")
    publish = (f"umask 077; mkdir -p {shlex.quote(root)}; test ! -e {shlex.quote(root+'/result.json')}; "
               f"cat > {shlex.quote(root+'/.result.json.partial')}; sync {shlex.quote(root+'/.result.json.partial')}; "
               f"mv {shlex.quote(root+'/.result.json.partial')} {shlex.quote(root+'/result.json')}")
    published = subprocess.run(ssh + [publish], input=(json.dumps(result,sort_keys=True)+"\n").encode(), capture_output=True, timeout=30)
    if published.returncode != 0:
        raise ModelSmokeError("could not publish model smoke result")
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True); p.add_argument('--url-file',type=Path,required=True)
    p.add_argument('--worker-host',required=True); p.add_argument('--worker-port',type=int,required=True)
    p.add_argument('--worker-key',type=Path,required=True); p.add_argument('--worker-result-root',required=True)
    a=p.parse_args(); spec=json.loads(a.spec.read_text())
    print(json.dumps(run(spec,url_file=a.url_file,worker_host=a.worker_host,worker_port=a.worker_port,worker_key=a.worker_key,worker_result_root=a.worker_result_root),sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
