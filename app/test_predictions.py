from PIL import Image
from prediction import pred_results

def main():
    img_path = "./data/deployement/full_images/img_8918.png"
    model_path = "./model/model3/model_v5.keras"  
    patient_id = "8918"

    input_image = Image.open(img_path)
    results = pred_results(input_image, model_path, patient_id)

    for pred, name in results:
        print(pred, name)

if __name__ == "__main__":
    main()
