from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import time
import aiohttp
import asyncio
import subprocess
import uuid
import threading
import GPUtil
from birefnet_processor import BiRefNetProcessor
from PIL import Image,ImageChops

app = FastAPI()

birefnet_processor = BiRefNetProcessor(gpu_id=0)

class TrainRequest(BaseModel):
    name: str
    prompt: str

task_map = {}

def process_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    
    bbox = ImageChops.difference(img, Image.new('RGBA', img.size, (0, 0, 0, 0))).getbbox()
    if bbox:
        img = img.crop(bbox)
    
    width, height = img.size
    target_size = 1024
    
    target_area = (target_size * target_size) * (1/2)
    current_area = width * height
    scale = (target_area / current_area) ** 0.5
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    if new_width > target_size or new_height > target_size:
        scale = target_size / max(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    new_img = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    new_img.paste(img, (paste_x, paste_y), img)
    
    new_img.save(image_path, "PNG")

@app.get("/is_alive")
async def is_alive():
    return JSONResponse(content={"status": "alive"}, status_code=200)

@app.post("/train")
async def train_lora(
    name: str = Form(...),
    prompt: str = Form(...),
    video_url: str = Form(...)
):
    input_dir = f"./input/{name}"
    os.makedirs(input_dir, exist_ok=True)
    
    train_dir = f"./train_data/{name}/10_data"
    os.makedirs(train_dir, exist_ok=True)

    video_path = f"{input_dir}/{name}.mp4"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(video_url) as response:
                if response.status == 200:
                    with open(video_path, "wb") as f:
                        f.write(await response.read())
                else:
                    raise HTTPException(status_code=400, detail=f"无法下载视频: {video_url}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"下载视频时出错: {str(e)}")

    try:
        duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}"
        duration = float(subprocess.check_output(duration_cmd, shell=True).decode('utf-8').strip())
        
        interval = duration / 8
        
        for i in range(8):
            time_point = i * interval
            output_path = f"{train_dir}/{i+1}.png"
            ffmpeg_cmd = f"ffmpeg -i {video_path} -ss {time_point} -frames:v 1 {output_path}"
            subprocess.run(ffmpeg_cmd, shell=True, check=True)
            
            birefnet_processor.extract_object(output_path).save(output_path)
            process_image(output_path)
            
            txt_path = f"{train_dir}/{i+1}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"处理视频时出错: {str(e)}")

    gpus = GPUtil.getGPUs()
    if not gpus:
        raise HTTPException(status_code=500, detail="没有可用的GPU")
    
    selected_gpu = min(gpus, key=lambda gpu: gpu.memoryUsed)
    gpu_id = selected_gpu.id

    task_id = str(uuid.uuid4())
    
    cmd = f"conda run -n lora-scripts CUDA_VISIBLE_DEVICES={gpu_id} ./lora-scripts/train_ar.sh {name}"
    print(cmd)

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