from fastapi import FastAPI, UploadFile, File, Body
from typing import Dict
from PIL import Image
import io
import os
from fastapi.responses import JSONResponse, StreamingResponse
from app.prediction import pred_results
from app.heatmap import construct_prediction_heatmap, construct_real_heatmap
from app.merge import merge_images
import json 

app = FastAPI()

@app.get('/status')
def status():
    return {"status" : "running"}

@app.post('/')
async def predict_image(file: UploadFile = File(...) ):
    try:
        print("received request")
        contents = await file.read()
        print("read file into memory")

        image = Image.open(io.BytesIO(contents))
        print("image converted")

        print(os.path.exists('../model/model3/model_v5.keras'))
        model_path = './model/model3/model_v5.keras'
        patient_id = (file.filename.split("_")[1]).split(".")[0]
        print(f"patient id is {id}")
        
        predictions = pred_results(image, model_path, patient_id)
        print("predictions computed")

        formatted = [
                {"patch": name, "prediction": pred.tolist()}  # pred is a NumPy array
                for pred, name in predictions
            ]
        print("predicted formated to JSON")

        predicted = construct_prediction_heatmap(original_image=image,predictions=formatted)
        print("heatmap constructed")
        real = construct_real_heatmap(patient_id=patient_id)
        print("real heatmap constructed")
        final_image = merge_images(predicted, real)
        print("images merged")
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f'error : {e}')
        return JSONResponse(status_code=500, content={"error":str(e)})



        