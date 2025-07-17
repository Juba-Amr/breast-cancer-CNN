from PIL import Image, ImageDraw
import json
import os
import matplotlib.cm as cm
import matplotlib as mpl

def construct_prediction_heatmap(original_image, predictions, output_path="data/clean/heatmaps/heatmap_result.png", cmap_name="jet"):
    os.makedirs("data/clean/heatmaps/", exist_ok=True)
    image = Image.open(original_image).convert("RGBA") 
    width, lenght = image.size
    overlay = Image.new("RGBA",(width,lenght))
    draw = ImageDraw.Draw(overlay)

    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    with open(predictions, 'r') as f:
        preds = json.load(f)["predictions"]
    for item in preds:
        patch_name = item["patch"] #eg 8918_x50_y600.png
        pred_value = item["prediction"][0]

        #we get the coords
        try:
            x = int((patch_name.split("_")[1])[1:])
            y = int((patch_name.split("_")[2])[1:])
        except:
            continue
        
        rgba_float = cmap(norm(pred_value))
        rgba = tuple(int(255*c) for c in rgba_float)
        print(rgba)
        draw.rectangle([x, y, x+50, y+50], fill=rgba)
    #print("image :", image.size)
    #print("overlay:", overlay.size)
    result = Image.alpha_composite(image, overlay)
    result.save(output_path)
    f"Saved heatmap to {output_path}"
    result.show()
    return result 


def construct_real_heatmap(patient_id):
    
    return None