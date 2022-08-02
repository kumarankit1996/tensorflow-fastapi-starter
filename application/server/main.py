import uvicorn
import logging
logging.basicConfig(level=logging.INFO,format='%(process)d-%(levelname)s-%(message)s')

from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from application.components import predict, read_imagefile
import timeit

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<br>by Ankit Kumar Sinha
<br> Ref: https://github.com/aniketmaurya/tensorflow-fastapi-starter-pack"""

app = FastAPI(title='Tensorflow FastAPI Starter', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    logging.info(f"Reading the image started")
    start = timeit.default_timer()
    image = read_imagefile(await file.read())
    logging.info(f"Image reading completed in: {timeit.default_timer()-start} sec")

    logging.info("Model Prediction started")
    start = timeit.default_timer()
    prediction = predict(image)
    logging.info(f"Model Prediction completed in: {timeit.default_timer()-start} sec")
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
