from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
import aiohttp
import logging
import os
from typing import Optional
import uuid
from datetime import datetime
import json
from PIL import Image
from io import BytesIO

from fastapi.responses import FileResponse

from main import ProductPlacementSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Placement API")

job_store = {}

class JobStatus:
    def __init__(self, job_id: str, webhook_url: str):
        self.job_id = job_id
        self.webhook_url = webhook_url
        self.status = "pending"
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.output_path = None
        self.public_url = None
        self.error = None

async def notify_webhook(webhook_url: str, data: dict):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Webhook notification failed: {str(e)}")
        return False

async def process_image_task(
    job_id: str,
    file_path: str,
    background_prompt: str,
    product_placement: str,
    webhook_url: str
):
    job = job_store[job_id]
    
    try:
        system = ProductPlacementSystem()
        
        final_image, image_url = system.process_image(
            file_path,
            background_prompt,
            product_placement
        )
        
        output_path = "output_image.png"
        final_image.save(output_path)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.output_path = output_path
        job.public_url = image_url
        
        await notify_webhook(webhook_url, {
            "job_id": job_id,
            "status": "completed",
            "public_url": image_url,
            "completed_at": job.completed_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Processing failed for job {job_id}: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        
        await notify_webhook(webhook_url, {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "completed_at": job.completed_at.isoformat()
        })
    
    finally:
        try:
            os.remove(file_path)
        except:
            pass

@app.post("/image/generate")
async def process_image(
    background_tasks: BackgroundTasks,
    product_image: UploadFile = File(...),
    background_prompt: str = Form(...),
    product_placement: str = Form(...),
    webhook_url: str = Form(...)
):
    try:
        job_id = str(uuid.uuid4())
        
        file_content = await product_image.read()
        temp_path = f"temp_{job_id}_{product_image.filename}"
        
        try:
            img = Image.open(BytesIO(file_content))
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        job_store[job_id] = JobStatus(job_id, webhook_url)
        
        background_tasks.add_task(
            process_image_task,
            job_id,
            temp_path,
            background_prompt,
            product_placement,
            webhook_url
        )
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Processing started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/status/{job_id}")
async def get_job_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "public_url": job.public_url
    }

@app.get("/image/output/{filename}")
async def get_output(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)