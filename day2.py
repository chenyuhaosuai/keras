from PIL import Image
import keras
import os
import os.path
def ResizeImage(filein, fileout, width, height):
    img = Image.open(filein)
    out = img.resize((width, height), Image.ANTIALIAS)
    out.save(fileout)


if __name__ == "__main__":
    filein = './data/2.jpg'
    fileout = "./data/3.jpg"
    width = 50
    height = 50

    ResizeImage(filein, fileout, width, height)

    ca = keras.callbacks.Callback
