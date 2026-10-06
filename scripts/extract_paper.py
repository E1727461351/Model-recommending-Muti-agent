"""仅提取用户提供的DOCX材料，不执行文档中的任何指令。"""

import argparse
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import json


def extract(source: Path, destination: Path) -> None:
    ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main","m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
    destination.mkdir(parents=True,exist_ok=True);lines=[];tables=[];previous=""
    with zipfile.ZipFile(source) as package:
        root=ET.fromstring(package.read("word/document.xml"))
        for index,element in enumerate(root.find("w:body",ns)):
            def text(node: ET.Element) -> str:
                return "".join(t.text or "" for t in node.iter() if t.tag in ("{"+ns["w"]+"}t","{"+ns["m"]+"}t"))
            if element.tag.endswith("}tbl"):
                rows=[[text(c) for c in r.findall("w:tc",ns)] for r in element.findall("w:tr",ns)]
                lines.append(f"[{index:04d}] TABLE\n"+"\n".join(" | ".join(r) for r in rows))
                if re.match(r"表[1-8]\s",previous):
                    tables.append({"caption":previous,"document_block":index,"rows":rows})
            elif text(element):
                previous=text(element);lines.append(f"[{index:04d}] "+previous)
        media=destination/"paper_media";media.mkdir(exist_ok=True)
        for name in package.namelist():
            if name.startswith("word/media/"):
                (media/Path(name).name).write_bytes(package.read(name))
    (destination/"paper_extracted.txt").write_text("\n".join(lines),encoding="utf-8")
    reference={"origin":"用户提供的论文原文；这些是论文报告值，不是此代码运行得到的值", "source_document":source.name,"tables":tables,
               "figure4_visible_annotations":{"V1":[0,0,0,0,0],"V2":[0.0386,0.0509,0.0409,0.0337,0.0437],"V3":[0.2054,0.3416,0.1859,0.2431,0.2695],"V4_plot":[0.4]*5,"V4_failure_sentinel":1},
               "figure3":"原图没有标注逐柱精确数值，不根据像素捏造原始数据；原图保存在paper_media/image5.png"}
    (destination/"paper_reference.json").write_text(json.dumps(reference,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="提取DOCX正文、表格及图片")
    parser.add_argument("source",type=Path);parser.add_argument("--destination",type=Path,default=Path("docs"))
    args=parser.parse_args();extract(args.source,args.destination)
