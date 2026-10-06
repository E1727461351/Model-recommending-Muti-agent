"""依据显式列映射转换数据，不自行猜测标签或划分。"""

import csv
from pathlib import Path
from typing import Dict
import numpy as np
from scipy.io import loadmat
from .io import read_json


def convert(mapping_path: Path, destination: Path) -> None:
    mapping=read_json(mapping_path);output={}
    for split in ("train","val","test"):
        spec=mapping["splits"][split]
        path=Path(spec["path"])
        if not path.is_absolute():
            path=Path(mapping_path).parent/path
        if path.suffix.lower()==".csv":
            with path.open(encoding=spec.get("encoding","utf-8-sig"),newline="") as stream:
                rows=list(csv.DictReader(stream))
            x=np.array([[float(r[k]) for k in spec["x_columns"]] for r in rows],dtype=np.float32)
            y=np.array([float(r[spec["y_column"]]) for r in rows])
            if spec.get("group_column"):
                output["groups_"+split]=np.array([r[spec["group_column"]] for r in rows])
        else:
            if path.suffix.lower()==".mat":
                values=loadmat(path)
            elif path.suffix.lower()==".npz":
                with np.load(path,allow_pickle=False) as loaded:
                    values={k:loaded[k] for k in loaded.files}
            else:
                raise ValueError("转换器支持CSV、MAT、NPZ；特殊目录结构请先用数据提供方脚本转换")
            x,y=np.asarray(values[spec["x_key"]],dtype=np.float32),np.asarray(values[spec["y_key"]]).reshape(-1)
            if "axis_order" in spec:
                x=np.transpose(x,spec["axis_order"])
            if spec.get("group_key"):
                output["groups_"+split]=np.asarray(values[spec["group_key"]]).reshape(-1).astype(str)
        if spec.get("sample_shape"):
            x=x.reshape((len(x),*spec["sample_shape"]))
        if len(x)!=len(y):
            raise ValueError("特征与标签数量不一致")
        output["X_"+split],output["y_"+split]=x,y
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError("目标数据已存在，避免覆盖请使用新路径")
    np.savez_compressed(destination,**output)
