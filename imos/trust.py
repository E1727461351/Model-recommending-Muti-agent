"""来源签名、关键制品完整性与滑动窗口运行可信度。"""

import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from .io import digest


def safe_path(root: Path, relative: str) -> Path:
    path = (Path(root)/relative).resolve()
    if Path(root).resolve() not in path.parents:
        raise ValueError("模型制品路径必须位于 modelbase 内")
    return path


def signed_payload(model: Dict) -> bytes:
    fields = {k:model.get(k) for k in ("id","publisher","source","version","architecture","config","preprocess","dependencies","baseline_hashes","kind","dataset","task","input_dim","output_dim","environment")}
    return json.dumps(fields,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")


def seal(model: Dict, root: Path, key_file: Path) -> Dict:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = load_pem_private_key(Path(key_file).read_bytes(),password=None)
    if not isinstance(key,Ed25519PrivateKey):
        raise ValueError("仅支持 Ed25519 私钥")
    model["baseline_hashes"] = {p:digest(safe_path(root,p)) for p in model["artifacts"]}
    model["signature"] = base64.b64encode(key.sign(signed_payload(model))).decode("ascii")
    return model


def trust_values(model: Dict, root: Path, config: Dict, history: List[Dict], now: float = None) -> Tuple[float,float,float]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    now = time.time() if now is None else now
    anchors = config.get("public_keys",{})
    publisher = model.get("publisher")
    s1 = int(publisher in anchors and bool(model.get("source")))
    s2 = 0
    if s1:
        try:
            key = Ed25519PublicKey.from_public_bytes(base64.b64decode(anchors[publisher],validate=True))
            key.verify(base64.b64decode(model.get("signature",""),validate=True),signed_payload(model))
            s2 = 1
        except (ValueError,InvalidSignature):
            s2 = 0
    s3 = int(bool(model.get("version")) and bool(model.get("source")) and bool(model.get("baseline_hashes")))
    src = float(sum(w*s for w,s in zip(config["source_weights"],(s1,s2,s3))))
    hashes = model.get("baseline_hashes",{})
    intact = bool(hashes) and set(hashes) == set(model.get("artifacts",[]))
    for relative,expected in hashes.items():
        path = safe_path(root,relative)
        intact = intact and path.is_file() and digest(path) == expected
    recent = [r for r in history if r["model"] == model["id"] and now-config["history_window_s"] <= r["time"] <= now]
    if not recent:
        run = config["cold_start_runtime_trust"]
    else:
        # 一次调用只记一个已归因的模型异常，环境或数据根因不扣模型分
        kinds = ("timeout","output","fail","resource","other")
        penalty = sum(w*sum(r.get("model_fault")==k for r in recent)/len(recent) for w,k in zip(config["runtime_weights"],kinds))
        run = max(0.0,min(1.0,1.0-penalty))
    return src,float(intact),float(run)
