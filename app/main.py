from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from fastapi.responses import JSONResponse
from prediction import pred_results

app = FastAPI()


@app.post('/')
async def predict_image(file: UploadFile = File() ):
    try:
        print("received request")
        contents = await file.read()
        print("read file into memory")

        image = Image.open(io.BytesIO(contents))
        print("image converted")

        model_path = '../model/model3/model_v5.keras'
        patient_id = file.filename.split("_")[1]
        
        predictions = pred_results(image, model_path, patient_id)
        print("predictions computed")

        formatted = [
                {"patch": name, "prediction": pred.tolist()}  # pred is a NumPy array
                for pred, name in predictions
            ]

        return JSONResponse(content={"predictions": formatted})
    except Exceptions as e:
        print(f'error : {e}')
        return JSONResponse(status_code=500, content={"error":str(e)})

