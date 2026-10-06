"""生成本地签名私钥并显示供管理员明确信任的公钥。"""

import argparse
import base64
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="创建模型发布者签名密钥；私钥不可提交Git")
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    if args.destination.exists():
        raise FileExistsError("不覆盖现有私钥")
    key=Ed25519PrivateKey.generate()
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    args.destination.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    print("请由模型库管理员确认发布者身份后，把以下公钥加入配置的public_keys：")
    print(base64.b64encode(public).decode("ascii"))
