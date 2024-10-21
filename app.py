from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from typing import List
import os
import time
import aiohttp
import asyncio
import subprocess
import uuid
import threading
import GPUtil

app = FastAPI()

class TrainRequest(BaseModel):
    name: str
    urls: List[str]
    prompt: str

task_map = {}

@app.post("/train")
async def train_lora(
    name: str = Form(...),
    prompt: str = Form(...),
    images: str = Form(...) 
):
    base_dir = f"./train_data/{name}"
    # timestamp = int(time.time())
    train_dir = f"{base_dir}/10_data"
    os.makedirs(train_dir, exist_ok=True)

    image_urls = images.split(',')
    image_urls = [url.lstrip('@') for url in image_urls]

    async def download_image(session, url, index):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_path = f"{train_dir}/{index}.jpg"
                    with open(image_path, "wb") as f:
                        f.write(await response.read())
                    
                    txt_path = f"{train_dir}/{index}.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(prompt)
                else:
                    raise HTTPException(status_code=400, detail=f"无法下载图片: {url}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"下载图片时出错: {str(e)}")

    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url, i+1) for i, url in enumerate(image_urls)]
        await asyncio.gather(*tasks)

    gpus = GPUtil.getGPUs()
    if not gpus:
        raise HTTPException(status_code=500, detail="没有可用的GPU")
    
    selected_gpu = min(gpus, key=lambda gpu: gpu.memoryUsed)
    gpu_id = selected_gpu.id

    task_id = str(uuid.uuid4())
    
    cmd = f"conda run -n lora-scripts CUDA_VISIBLE_DEVICES={gpu_id} ./lora-scripts/train_ar.sh {name}"
    print(cmd)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def run_process(cmd):
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        print(f"进程退出，返回码: {process.returncode}")

    thread = threading.Thread(target=run_process, args=(cmd,))
    thread.start()

    task_map[task_id] = {
        "name": name,
        "thread": thread,
        "start_time": time.time()
    }

    return {"message": "训练任务已启动", "task_id": task_id, "train_dir": train_dir, "selected_gpu": gpu_id}

@app.post("/task")
async def get_task_status(task_id: str = Form(...)):
    if task_id not in task_map:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = task_map[task_id]
    thread = task["thread"]
    
    if thread.is_alive():
        status = "运行中"
    else:
        status = "已完成"
    
    return {
        "task_id": task_id,
        "name": task["name"],
        "status": status,
        "start_time": task["start_time"],
        "elapsed_time": time.time() - task["start_time"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)