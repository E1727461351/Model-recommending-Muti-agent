"""统一数据契约与同工序工况变化构造，不假定原始数据文件格式。"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy import signal


def load_dataset(root: Path, scene: str) -> Dict[str, np.ndarray]:
    path = Path(root) / scene / "data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}，请按 datasets/README.md 转换真实数据并显式提供数据划分")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    for split in ("train", "val", "test"):
        x, y = data.get("X_"+split), data.get("y_"+split)
        if x is None or y is None or x.ndim not in (2, 3) or len(x) != len(y) or not len(x):
            raise ValueError(f"{scene} 的 {split} 缺失或形状不符合约定")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{scene} 的 {split} 存在缺失值或无穷值")
    # 可选的物理试件ID用于检验跨集合泄漏
    if all("groups_"+s in data for s in ("train", "val", "test")):
        groups = [set(data["groups_"+s].tolist()) for s in ("train", "val", "test")]
        if any(groups[a] & groups[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError("同一试件或寿命轨迹出现在多个划分中")
    return data


def prepare_signal(x: np.ndarray, config: Dict) -> np.ndarray:
    if x.ndim == 2:
        return x.astype(np.float32)
    # 频带与采样频率由数据拥有者提供，不从论文推测
    band, fs = config.get("bandpass_hz"), config.get("sampling_rate_hz")
    result = x
    if band:
        if not fs or not 0 < band[0] < band[1] < fs/2:
            raise ValueError("带通频率必须位于采样频率的奈奎斯特范围内")
        sos = signal.butter(config.get("filter_order", 4), band, fs=fs, btype="bandpass", output="sos")
        result = signal.sosfiltfilt(sos, result, axis=1)
    factor = config.get("downsample", 1)
    if factor > 1:
        result = signal.resample_poly(result, 1, factor, axis=1)
    return result.astype(np.float32)


def statistical_features(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.astype(np.float32)
    mean, std = x.mean(1), x.std(1)
    rms = np.sqrt((x*x).mean(1))
    peak = np.max(np.abs(x), axis=1)
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    centroid = (spectrum * np.arange(spectrum.shape[1])[None,:,None]).sum(1) / np.maximum(spectrum.sum(1), 1e-8)
    return np.concatenate([mean, std, rms, peak, centroid/max(spectrum.shape[1]-1,1)], axis=1).astype(np.float32)


def perturb(x: np.ndarray, size_ratio: float, amplitude_ratio: float) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("尺寸扰动要求形状为 样本×时间×通道 的时序输入")
    if size_ratio <= 0 or amplitude_ratio <= 0:
        raise ValueError("扰动比例必须为正数")
    # 重建假设：固定长度内按尺寸比例伸缩时间，边界周期延拓
    t = np.arange(x.shape[1], dtype=float)
    query = t / size_ratio
    out = np.empty_like(x)
    for i in range(len(x)):
        for c in range(x.shape[2]):
            out[i,:,c] = np.interp(query, t, x[i,:,c], period=x.shape[1]) * amplitude_ratio
    return out


def build_cross_scenes(root: Path, config: Dict) -> None:
    data = load_dataset(root, "D2-a")
    for name in ("D2-b", "D2-c"):
        spec = config["scenarios"][name]
        out = dict(data)
        for split in ("train", "val", "test"):
            out["X_"+split] = perturb(data["X_"+split], spec["size_ratio"], spec["amplitude_ratio"])
        directory = Path(root)/name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/"data.npz", **out)
