"""实时资源快照与有界物理资源占用，默认实验使用可用配额注入。"""

import multiprocessing as mp
import time
from typing import Dict, List
import psutil
import torch


def snapshot() -> Dict:
    memory=psutil.virtual_memory()
    out={"cpu_count":psutil.cpu_count(logical=False) or 1,"cpu_usage_percent":psutil.cpu_percent(interval=None),
         "available_ram_gb":memory.available/1024**3,"total_ram_gb":memory.total/1024**3}
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        out.update({"gpu_name":torch.cuda.get_device_name(),"available_gpu_gb":free/1024**3,"total_gpu_gb":total/1024**3})
    return out


def burn_cpu(stop: mp.Event) -> None:
    while not stop.is_set():
        value=1.001
        for _ in range(20000):
            value=value*1.000001%100000


class PhysicalPressure:
    def __init__(self, cpu_fraction: float = 0.5, retained_ram_gb: float = 2.0):
        self.cpu_fraction=cpu_fraction;self.retained=retained_ram_gb
        self.stop=mp.Event();self.processes: List[mp.Process]=[];self.ram=[];self.gpu=None

    def __enter__(self):
        for _ in range(max(1,int((psutil.cpu_count(logical=True) or 1)*self.cpu_fraction))):
            process=mp.Process(target=burn_cpu,args=(self.stop,),daemon=True);process.start();self.processes.append(process)
        # 分块占用并实际触碰内存，保留要求的RAM；操作系统缓存/其他进程会使值浮动
        chunk=32*1024**2
        while psutil.virtual_memory().available-self.retained*1024**3>chunk:
            block=bytearray(chunk)
            for index in range(0,chunk,4096):
                block[index]=1
            self.ram.append(block)
        if torch.cuda.is_available():
            free,_=torch.cuda.mem_get_info()
            # CUDA上下文需保留少量显存，无法安全声称占用数学意义上的全部显存
            count=max(0,(free-256*1024**2)//4)
            self.gpu=torch.empty(int(count),dtype=torch.float32,device="cuda")
            self.gpu.fill_(1)
        return self

    def __exit__(self,exc_type,exc,tb):
        self.stop.set()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate();process.join(timeout=2)
        self.ram.clear();self.gpu=None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="单独进行有界物理资源压力测试，退出时释放资源")
    parser.add_argument("--seconds",type=float,default=15)
    args=parser.parse_args()
    if not 0<args.seconds<=60:
        raise ValueError("持续时间必须在0到60秒之间")
    print("占用前",snapshot())
    with PhysicalPressure():
        print("占用中",snapshot());time.sleep(args.seconds)
    print("释放后",snapshot())
