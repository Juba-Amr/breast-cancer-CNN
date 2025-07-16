from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from fastapi.responses import JSONResponse
from prediction import pred_results

app = FastAPI()


@app.POST('/')
async def predict_image(file: UploadFile = File() ):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    model_path = '../mode/model3/model_v5.keras'
    patient_id = file.filename.split("_")[1]
    
    predictions = pred_results(image, model_path, patient_id)

    formatted = [
            {"patch": name, "prediction": pred.tolist()}  # pred is a NumPy array
            for pred, name in predictions
        ]

    return JSONResponse(content={"predictions": formatted})


