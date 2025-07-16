from PIL import Image
import numpy as np

def crop_slice_image(image, id):
    images=[]
                                        
    img_sample = image.convert("RGB")
    img_w, img_h = img_sample.size

    img_sample = img_sample.crop((1,1,img_w , img_h ))
    
    for x in range(0,img_w,50):
        for y in range(0,img_h,50):
            patch = img_sample.crop((x,y,x+50,y+50))
            patch = patch.convert('RGB')

            name = f'{id}_x{x}_y{y}'

            patch = np.array(patch)
            if not np.all(patch==255) and not np.all(patch==0):
                images.append((patch,name))
    return images

