from app.prediction import pred_results
from app.merge import merge_images
from PIL import Image


def test_pred_results():
    image_path = open("unit_tests/image/img_9225.png", "rb")
    input_image = Image.open(image_path)
    model_path="model/model3/model_v5.keras"
    patient_id=9225

    preds = pred_results(input_image,model_path,patient_id) 
    output_y = [y for y,_ in preds]

    assert len(output_y)>0
    for y in output_y:
        assert 0<= y <=1

def test_merge_images_returns_image():
    from PIL import Image
    dummy = Image.new("RGB", (100, 100))
    result = merge_images(dummy, dummy)
    assert isinstance(result, Image.Image)
    assert result.size == (2*100, 100+500)
