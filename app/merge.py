from PIL import Image, ImageDraw
import numpy as np

def merge_images(image1,image2):
    img1 = image1#Image.open(image1).convert("RGBA")
    img2 = image2#Image.open(image2).convert("RGBA")
    n=2

    width1, height1 = img1.size
    width2, height2 = img2.size

    max_width = width1+width2
    max_height = np.max([height1,height2]) 
    size = int(max_height*0.125)

    canva = Image.new("RGBA",(max_width,max_height + 500),color="white")
    canva.paste(img1,(0,0))
    canva.paste(img2,(width1,0))
    draw = ImageDraw.Draw(canva)
    draw.line([(width1,0),(width1,max_height+500)], fill="black", width=10)
    draw.line([(0,max_height),(max_width,max_height)], fill="black", width=10)
    draw.text((n*(width1//100),height1 + n*(height1//100)),text="Predictions Heatmap",fill="black", font_size=size)
    draw.text((width1 + n*(width2//100),height1 + n*(height2//100)),text="Real Heatmap",fill="black", font_size=size)

    canva.show()
    return canva