import pygame
from PIL import Image
from settings import *

SIZE_X = len(WORLD_MAP)
SIZE_Y = len(WORLD_MAP[0])

print("SIZE_X",SIZE_X)
print("SIZE_Y", SIZE_Y)

im1 = Image.open('../graphics/cave_images/ground.png').resize((64,64))


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst



img = im1
for i in range(SIZE_X-1):
    img = get_concat_h(img, im1)
    

img2 = img
for i in range(SIZE_Y-1):
    img2 = get_concat_v(img2, img)
    
img2.save('../graphics/ground_map.png')