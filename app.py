import cv2
import numpy as np
import time

from fastapi import FastAPI
from fastapi import Request

from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from fastapi.templating import Jinja2Templates

from fastapi import File, UploadFile

import detector as detector_x
import base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")

detector_x.start_detector()


# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):

#     return templates.TemplateResponse(name="index.html", request=request)


# @app.get("/", response_class=HTMLResponse)
# async def live_page(request: Request):

#     return templates.TemplateResponse(name="live.html", request=request)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "page": "home"}
    )


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "page": "upload"}
    )


@app.get("/live", response_class=HTMLResponse)
async def live_page(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "page": "live"}
    )


@app.get("/stats")
async def stats():

    return JSONResponse(detector_x.vehicle_count)


def generate_frames():

    while True:

        frame = detector_x.latest_frame

        if frame is None:

            time.sleep(0.05)

            continue

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.get("/video")
def video():

    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/health")
def health():

    return {"status": "ok"}


# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):

#     return templates.TemplateResponse(name="index.html", request=request)


# @app.get("/upload", response_class=HTMLResponse)
# async def upload_page(request: Request):

#     return templates.TemplateResponse(name="upload.html", request=request)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = detector_x.model.predict(source=img, imgsz=640, conf=0.35, verbose=False)

    result = results[0]

    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    if result.boxes is not None:

        for box in result.boxes:

            cls = int(box.cls[0])

            if cls < len(detector_x.CLASS_NAMES):

                counts[detector_x.CLASS_NAMES[cls]] += 1

    annotated = result.plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {"counts": counts, "image": img_base64}
