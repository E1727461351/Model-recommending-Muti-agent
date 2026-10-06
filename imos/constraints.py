"""硬件、软件、权限与性能约束的显式匹配。"""

from typing import Dict, List
from packaging.specifiers import SpecifierSet


def mismatch(required: Dict, current: Dict) -> List[str]:
    reasons=[]
    for k in ("gpu_memory_gb","cpu_cores","ram_gb","storage_gb"):
        if required.get(k,0) > current.get(k,0):
            reasons.append(k)
    for k in ("os","runtime","gpu_model"):
        value = required.get(k,"*")
        if value not in ("*",None) and current.get(k) != value:
            reasons.append(k)
    version = required.get("runtime_version","*")
    if version != "*" and (current.get("runtime_version","*")=="*" or current["runtime_version"] not in SpecifierSet(version)):
        reasons.append("runtime_version")
    for name,spec in required.get("dependencies",{}).items():
        value=current.get("dependencies",{}).get(name)
        if value is None or value not in SpecifierSet(spec):
            reasons.append("dependency:"+name)
    for k in ("auth_level","privacy_level"):
        if required.get(k,0)>current.get(k,0):
            reasons.append(k)
    if required.get("latency_ms",0)>current["max_latency_ms"]:
        reasons.append("latency")
    if required.get("availability",100)<current.get("min_availability",0):
        reasons.append("availability")
    return reasons


def constraint_fit(required: Dict, current: Dict) -> float:
    latency = required.get("latency_ms",0)
    budget = current["max_latency_ms"]
    slack = max(0.0,min(1.0,1-latency/max(budget,1e-8)))
    target = current.get("min_throughput",0)
    throughput = min(1.0,required.get("throughput",0)/target) if target else 1.0
    return (slack+throughput)/2
