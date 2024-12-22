from PIL import Image
from lib.tagging.joycaption import JoyCaptioning

if __name__=='__main__':
    path = '/home/lars/SD/datasets/lars/penis/IMG_7579.png'

    jc = JoyCaptioning()
    # read image at path as pillow Image
    print(jc.predict(Image.open(path)))

