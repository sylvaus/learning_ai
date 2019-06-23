from random import randint

import pygame
from pygame import Vector2, Surface

from pygame.sprite import Sprite, Group, spritecollide
from pygame.time import Clock

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TRANSPARENT = (0, 0, 0,)
WALL_IMAGE = pygame.image.load("images/wall.png")

def clamp_vec2(vec2: Vector2, min_vec2: Vector2, max_vec2: Vector2):
    return Vector2(
        max(min_vec2.x, min(max_vec2.x, vec2.x))
        , max(min_vec2.y, min(max_vec2.y, vec2.y))
    )


def clamp_abs_vec2(vec2: Vector2, max_abs_vec2: Vector2):
    return Vector2(
        max(-max_abs_vec2.x, min(max_abs_vec2.x, vec2.x))
        , max(-max_abs_vec2.y, min(max_abs_vec2.y, vec2.y))
    )


class Bird(Sprite):
    def __init__(
            self, center=None, speed=None, gravity=0.35, flap_accel=-10
            , max_abs_vel=None, *groups):
        super().__init__(*groups)
        self.image = pygame.image.load("images/bird.png")
        self.rect = self.image.get_rect()
        if center:
            self.rect.center = center

        self._pos = Vector2(self.rect.center)
        if speed is None:
            speed = Vector2(0, 0)
        self._speed = speed
        if max_abs_vel is None:
            max_abs_vel = Vector2(9.0, 9.0)
        self._max_abs_vel = max_abs_vel
        self._gravity = gravity
        self._flap_accel = flap_accel

    def flap(self):
        self._speed.y += self._flap_accel

    def update(self, *args):
        self._speed.y += self._gravity
        self._speed = clamp_abs_vec2(self._speed, self._max_abs_vel)
        self._pos += self._speed
        self.rect.center = self._pos


class Wall(Sprite):
    def __init__(self, center=None, speedx=-2, *groups):
        super().__init__(*groups)
        self.image = WALL_IMAGE
        self.rect = self.image.get_rect()
        if center:
            self.rect.center = center

        self._pos = Vector2(self.rect.center)
        self._speed = Vector2(speedx, 0)

    def update(self, *args):
        self._pos += self._speed
        self.rect.center = self._pos


class Border(pygame.sprite.Sprite):
    def __init__(self, size, center, color=BLACK, opacity=255, *groups):
        super().__init__(*groups)
        pygame.sprite.Sprite.__init__(self)
        self.image = Surface(size)
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = center


class BirdGame:
    def __init__(self, size=(800, 600), add_wall_intervals=None):
        pygame.init()
        self._screen_size = size
        self._screen = pygame.display.set_mode(size)
        self._bird = Bird((size[0]//10, size[1]//2))
        self._walls = Group()
        self._wall_list = []
        self._all_sprites = Group()
        self._all_sprites.add(self._bird)
        self._game_over = False
        if not add_wall_intervals:
            add_wall_intervals = list(range(125, 225))
        self._add_wall_intervals = add_wall_intervals

        self._add_wall_frame_left = self._add_wall_intervals[randint(0, len(self._add_wall_intervals) - 1)]
        self._add_wall()

    def is_game_over(self):
        return self._game_over

    def next_frame(self, key_up):
        if key_up:
            self._bird.flap()
        self._all_sprites.update()

        self._add_wall_if_needed()
        self._clean_sprites()
        self._draw()

    def _draw(self):
        self._screen.fill(WHITE)
        self._all_sprites.draw(self._screen)
        
        self._check_game_over()
        if self._game_over:
            self._display_game_over()

        pygame.display.flip()

    def _display_game_over(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Game Over', True, BLACK)
        text_rect = text.get_rect()
        text_rect.center = (self._screen_size[0] // 2, self._screen_size[1] // 2)
        self._screen.blit(text, text_rect)

    def _check_game_over(self):
        if self._bird.rect.bottom > self._screen_size[1]:
            self._game_over = True
        if self._bird.rect.top < 0:
            self._game_over = True

        if spritecollide(self._bird, self._walls, False):
            self._game_over = True

    def _add_wall(self):
        half_width = WALL_IMAGE.get_width() // 2
        half_height = WALL_IMAGE.get_height() // 2
        height = randint(150, 450)
        half_opening = 60

        top_wall = Wall((
            self._screen_size[0] + half_width
            , height - (half_opening + half_height)
        ))
        bottom_wall = Wall((
            self._screen_size[0] + half_width
            , height + (half_opening + half_height)
        ))

        self._walls.add(top_wall)
        self._walls.add(bottom_wall)
        self._wall_list.append(top_wall)
        self._wall_list.append(bottom_wall)
        self._all_sprites.add(top_wall)
        self._all_sprites.add(bottom_wall)

    def _add_wall_if_needed(self):
        self._add_wall_frame_left -= 1
        if self._add_wall_frame_left <= 0:
            self._add_wall_frame_left = self._add_wall_intervals[randint(0, len(self._add_wall_intervals) - 1)]
            self._add_wall()

    def _clean_sprites(self):
        if self._wall_list and self._wall_list[0].rect.right < 0:
            # Remove top and bottom walls
            self._walls.remove(self._wall_list.pop(0))
            self._walls.remove(self._wall_list.pop(0))


def main():
    game = BirdGame()
    clock = Clock()
    while not game.is_game_over():
        key_up = False
        quit_ = False
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                key_up = True
            if event.type == pygame.QUIT:
                quit_ = True

        if quit_:
            break

        game.next_frame(key_up)
        clock.tick(60)

    input("Press any key to exit.")


if __name__ == '__main__':
    main()
