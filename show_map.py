from settings import *
from PIL import Image



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

for i in range(len(WORLD_MAP)):
    for j in range(len(WORLD_MAP[i])):
       
        if WORLD_MAP[i][j] == 'x':
            current_tile = Image.open('graphics/rock.png')
        if WORLD_MAP[i][j] == ' ':
            current_tile = Image.open('graphics/ground.png')
        if WORLD_MAP[i][j] == 'p':
            current_tile = Image.open('graphics/player.png').resize((17,17))
        if WORLD_MAP[i][j] == 'end':
            current_tile = Image.open('graphics/out.png').resize((17,17))
        
        if j==0:
            img_to_construct_j = current_tile
            
        else:
            img_to_construct_j = get_concat_h(img_to_construct_j, current_tile)
            
        
    if i == 0:
        img_to_construct_i = img_to_construct_j
    else:
        img_to_construct_i = get_concat_v(img_to_construct_i, img_to_construct_j)
            
img_to_construct_i.save('graphics/visualize_map.png')