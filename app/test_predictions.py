from PIL import Image
from app.prediction import pred_results
from app.heatmap import construct_prediction_heatmap, construct_real_heatmap
from app.merge import merge_images

def main():
    img_path = "./data/deployement/full_images/img_8918.png"
    model_path = "./model/model3/model_v5.keras"  
    patient_id = "8918"
    
    preds = "data/testings/response/response.json"

    input_image = Image.open(img_path)
    #results = pred_results(input_image, model_path, patient_id)

    #for pred, name in results:
    #    print(pred, name)
    predhm =construct_prediction_heatmap(img_path, preds)
    realhm =construct_real_heatmap(patient_id)
    merge_images(predhm,realhm)

if __name__ == "__main__":
    main()
