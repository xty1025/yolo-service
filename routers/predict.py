from fastapi import APIRouter,File,UploadFile,HTTPException,Response
import numpy as np
import cv2
import ml
import threading
router = APIRouter()
GPU_LIMIT=threading.Semaphore(2)
def run_inference(data:bytes):
    buf=np.frombuffer(data,np.uint8)
    img = cv2.imdecode(buf,cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="不是有效图片")
    try:
        model = ml.get_model()
    except RuntimeError:
        raise HTTPException(status_code=503,detail="模型未就绪")
    with GPU_LIMIT:
        result = model.predict(img,conf=0.25,device='cpu')
    r=result[0]
    return img,r,model
@router.post('/predict')
def predict(file: UploadFile = File(...)):
    img,r,model = run_inference(file.file.read())
    detections=[]
    for box,p,c in zip(r.boxes.xyxy,r.boxes.conf,r.boxes.cls):
        detections.append({
            "class_id": int(c),
            "class_name": model.names[int(c)],
            "conf":float(p),
            "box":[float(v) for v in box]

        })

    return {"count":len(detections),"detections":detections}

@router.post('/predict/image')
def predict_image(file: UploadFile = File(...)):
    img,r,model = run_inference(file.file.read())
    for box,p,c in zip(r.boxes.xyxy,r.boxes.conf,r.boxes.cls):
        x1,y1,x2,y2 = [int(v) for v in box]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(img,f"{model.names[int(c)]} {float(p):.2f}",(x1,max(y1-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    ok,buf=cv2.imencode('.jpg',img)
    return Response(content=buf.tobytes(),media_type="image/jpeg")