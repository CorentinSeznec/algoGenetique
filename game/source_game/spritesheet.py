import pygame



class Spritesheet:
    def __init__(self, filename):
        self.filename = filename
        self.sprite_sheet = pygame.image.load(filename).convert()
        
        
    def get_sprite(self, x, y , w, h):
         sprite = pygame.Surface((w, h)) 
         sprite.blit(self.sprite_sheet, (0,0), (x, y, w ,h))
         return sprite
     
     
pygame.init()
DISPLAY_W, DISPLAY_H = 1280, 720
canvas = pygame.Surface((DISPLAY_W, DISPLAY_H ))
window = pygame.display.set_mode(((DISPLAY_W, DISPLAY_H)))
running = True

# mysprite = Spritesheet('../graphics/pokemon_spritesheet.png')
# tile_x = 20
# tile_y = 31
# tile_size = 17
# number_tiles = 1
# ground  = mysprite.get_sprite(tile_size*(tile_x-1), tile_size*(tile_y-1), tile_size*number_tiles, tile_size*number_tiles)
# ground = pygame.transform.scale(ground, (64*number_tiles, 64*number_tiles))
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         if event.type == pygame.KEYDOWN:
            
#             if event.key == pygame.K_SPACE:
#                 pass
            
#     canvas.fill((255, 255, 255))
#     canvas.blit(ground, (64, DISPLAY_H - 64 *(number_tiles+1)))
#     window.blit(canvas, (0,0))
#     pygame.display.update()
    
    
    
    
def sprite_sheet(size, file_name, pos=(0, 0)):
    """Loads a sprite sheet.
    (tuple(num, num), string, tuple(num, num) -> list

    This functions cuts the given image into pieces(sprites) and returns them as a list of images.
    WARNING!!! It needs the pygame library to be imported and initialised to work.
    Basically this piece of code is to be used preferably within an existing program.
    size is the size in pixels of the sprite. eg. (64, 64) sprite size of 64 pixels, by 64 pixels.
    file_name is the file name that contains the sprite sheet. preferable format is png.
    pos is the starting point on the image. eg. (10, 10) think about it as an offset.
    """
    len_sprt_x, len_sprt_y = size  # sprite size
    sprt_rect_x, sprt_rect_y = pos  # where to find first sprite on sheet
    sheet = pygame.image.load(file_name).convert_alpha()  # Load the sheet
    sheet_rect = sheet.get_rect()  # assign a rect of the sheet's size
    sprites = []  # make a list of sprites
    for i in range(0, sheet_rect.height, size[1]):  # rows
        for ii in range(0, sheet_rect.width, size[0]):  # columns
            sheet.set_clip(pygame.Rect(sprt_rect_x, sprt_rect_y, len_sprt_x, len_sprt_y))  # clip the sprite
            sprite = sheet.subsurface(sheet.get_clip())  # grab the sprite from the clipped area
            sprites.append(sprite)  # append the sprite to the list
            sprt_rect_x += len_sprt_x  # go to the next sprite on the x axis
        sprt_rect_y += len_sprt_y  # go to the next row (y axis)
        sprt_rect_x = 0  # reset the sprite on the x axis back to 0
    return sprites  # return the sprites


sprites = sprite_sheet((17,17), '../graphics/pokemon_spritesheet.png', pos=(0, 0))
   
  
# Importing Image module from PIL package 
from PIL import Image 
import PIL 
  
# creating a image object (main image) 
  
# save a image using extension
for index, sprite in enumerate(sprites):
    
    sprite.save(index+"tile.jpg")
