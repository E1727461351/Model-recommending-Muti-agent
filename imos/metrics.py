"""原文指标、原文矛盾项及补充的客观指标。"""

from typing import Any, Dict, List, Optional
import numpy as np


def performance(y: np.ndarray, pred: np.ndarray, task: str) -> Dict[str, Any]:
    y, pred = np.asarray(y).reshape(-1), np.asarray(pred).reshape(-1)
    if len(y) != len(pred) or not np.isfinite(pred).all() or not len(y):
        raise ValueError("预测长度不一致、预测非有限值或样本为空")
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        return {"accuracy": float(accuracy_score(y, pred)), "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))}
    error = pred - y
    variance = float(np.sum((y - y.mean()) ** 2))
    # 原文未说明零标签处理；出现零标签时报告空值，避免无依据地替换分母
    mape = float(np.mean(np.abs(error / y))) if np.all(y != 0) else None
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))), "MAPE": mape,
            "R2": float(1 - np.sum(error ** 2) / variance) if variance > 0 else None,
            "zero_labels": int(np.sum(y == 0))}


def quality(perf: Dict[str, Any], task: str) -> float:
    value = perf.get("accuracy" if task == "classification" else "R2")
    return float(np.clip(value if value is not None else 0.0, 0.0, 1.0))


def similarity(selected: Dict[str, Any], baseline: Dict[str, Any]) -> Optional[float]:
    keys = ["RMSE", "MAPE", "R2"]
    if any(selected.get(k) is None or baseline.get(k) is None for k in keys):
        return None
    return float(np.linalg.norm([selected[k] - baseline[k] for k in keys]))


def summarize(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trials)
    if not n:
        return {"status": "missing", "N": 0}
    successes = [r for r in trials if r.get("success")]
    sr_terms, cr_terms = [], []
    for r in trials:
        ok = bool(r.get("success"))
        retries = r.get("retries", 0)
        # 公式34与文字：失败或者重试十次以上记零
        sr_terms.append(max(0.0, 1.0 - 0.1 * retries) if ok and retries < 10 else 0.0)
        hit = ok and r.get("selected") == r.get("optimal")
        delta = r.get("quality_gap", 0.0)
        # 公式35的字面结果：未命中指示函数为零，加负差值可能得到负数
        cr_terms.append(1.0 if hit else -abs(delta) if ok and r.get("same_scene") else 0.0)
    ms = [r["MS"] for r in successes if r.get("MS") is not None]
    return {"N": n, "successful_N": len(successes), "SR_paper": 100 * float(np.mean(sr_terms)),
            "SR_fraction":float(np.mean(sr_terms)),
            "CR_paper_literal": 100 * float(np.mean(cr_terms)),
            "SR_binary": 100 * len(successes) / n,
            "CR_binary_success": 100 * sum(r.get("selected") == r.get("optimal") for r in successes) / len(successes) if successes else None,
            "CR_closeness_success": 100 * float(np.mean([max(0.0, 1.0-r.get("quality_gap", 0.0)) if r.get("same_scene") else 0.0 for r in successes])) if successes else None,
            "MS": float(np.mean(ms)) if ms else None,
            "MS_plot": float(np.mean(ms)) if ms else 1.0,
            "elapsed_all_s": float(np.mean([r["elapsed_s"] for r in trials])),
            "elapsed_success_s": float(np.mean([r["elapsed_s"] for r in successes])) if successes else None,
            "metric_warning": "CR原文公式与取值范围冲突；保留字面公式并独立报告客观命中率。MS失败为空，绘图哨兵为1。"}
