import pygame 
from settings import *



class Tile(pygame.sprite.Sprite):
	def __init__(self,pos,groups, col):
		super().__init__(groups)

		if col == 'x':
			self.image = pygame.image.load('../graphics/cave_images/rock.png').convert_alpha()

		elif col == "end" :
			self.image = pygame.image.load('../graphics/cave_images/out.png').convert_alpha()
				
		
  
		# if col == ' ':
		# 	self.image = pygame.image.load('../graphics/cave_images/ground.png').convert_alpha()
   

		self.image  = pygame.transform.scale(self.image, (64, 64))
  
		self.rect = self.image.get_rect(topleft = pos)
		self.hitbox = self.rect.inflate(-20,-20)