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

    @property
    def speed(self):
        return self._speed

    def flap(self):
        self._speed.y += self._flap_accel

    def update(self, *args):
        self._speed.y += self._gravity
        self._speed = clamp_abs_vec2(self._speed, self._max_abs_vel)
        self._pos += self._speed
        self.rect.center = self._pos


class Wall:
    def __init__(self, opening_height, opening_size, scren_width):
        half_width = WALL_IMAGE.get_width() // 2
        half_height = WALL_IMAGE.get_height() // 2

        self._opening_height = opening_height
        self._opening_top = opening_height - (opening_size // 2)
        self._opening_bottom = opening_height + (opening_size // 2)

        self._top_wall = WallSprite((
            scren_width + half_width
            , opening_height - (opening_size // 2 + half_height)
        ))
        self._bottom_wall = WallSprite((
            scren_width + half_width
            , opening_height + (opening_size // 2 + half_height)
        ))

    @property
    def opening_height(self):
        return self._opening_height

    @property
    def opening_top(self):
        return self._opening_top

    @property
    def opening_bottom(self):
        return self._opening_bottom

    @property
    def right(self):
        return self._top_wall.rect.right

    @property
    def left(self):
        return self._top_wall.rect.left

    @property
    def center(self):
        return self._top_wall.rect.center

    @property
    def sprites(self):
        return self._top_wall, self._bottom_wall

    def does_collide(self, sprite):
        if ((self.left <= sprite.rect.right <= self.right)
            or (self.left <= sprite.rect.left <= self.right))\
           and not ((self.opening_top < sprite.rect.top)
                    and (self.opening_bottom > sprite.rect.bottom)):
            return True
        return False


class WallSprite(Sprite):
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
    OPENING_HEIGHT = 120

    def __init__(self, size=(800, 600), add_wall_intervals=None, headless=False):
        self._headless = headless
        self._screen_size = size

        if not self._headless:
            pygame.init()
            self._screen = pygame.display.set_mode(size)
            self._score_font = pygame.font.Font('freesansbold.ttf', 20)

        if not add_wall_intervals:
            add_wall_intervals = list(range(125, 225))
        self._add_wall_intervals = add_wall_intervals

        self.reset()

    def reset(self):
        self._bird = Bird((self._screen_size[0] // 10, self._screen_size[1] // 2))
        self._walls = Group()
        self._wall_list = []
        self._all_sprites = Group()
        self._all_sprites.add(self._bird)
        self._game_over = False
        self._score = 0
        self._add_wall_frame_left = self._add_wall_intervals[randint(0, len(self._add_wall_intervals) - 1)]
        self._add_wall()

        self._active_wall = self._wall_list[0]
        if not self._headless:
            self._draw()

    @property
    def score(self):
        return self._score

    @property
    def distance_to_wall(self):
        return (
            self._active_wall.center[0] - self._bird.rect.centerx,
            self._active_wall.opening_height - self._bird.rect.centery
        )

    @property
    def bird_center(self):
        return self._bird.rect.center

    @property
    def bird_speed(self):
        return self._bird.speed

    def is_game_over(self):
        return self._game_over

    def next_frame(self, key_up):
        if key_up:
            self._bird.flap()
        self._all_sprites.update()

        self._add_wall_if_needed()
        self._update_active_wall()
        self._clean_sprites()
        self._check_game_over()

        if not self._game_over:
            self._score += 1/10

        if not self._headless:
            self._draw()

    def _draw(self):
        self._screen.fill(WHITE)
        self._all_sprites.draw(self._screen)
        self._display_score()
        if self._game_over:
            self._display_game_over()

        pygame.display.flip()

    def _display_score(self):
        text = self._score_font.render('Score: {}'.format(int(self._score)), True, BLACK)
        text_rect = text.get_rect()
        text_rect.top = 10
        text_rect.left = 10
        self._screen.blit(text, text_rect)

    def _display_game_over(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Game Over', True, BLACK)
        text_rect = text.get_rect()
        text_rect.center = (self._screen_size[0] // 2, self._screen_size[1] // 2)
        self._screen.blit(text, text_rect)

    def _check_game_over(self):
        if self._bird.rect.bottom > self._screen_size[1]:
            self._game_over = True

        if self._active_wall and self._active_wall.does_collide(self._bird):
            self._game_over = True

    def _add_wall(self):
        wall = Wall(randint(150, 450), self.OPENING_HEIGHT, self._screen_size[0])

        self._walls.add(*wall.sprites)
        self._wall_list.append(wall)
        self._all_sprites.add(*wall.sprites)

    def _add_wall_if_needed(self):
        self._add_wall_frame_left -= 1
        if self._add_wall_frame_left <= 0:
            self._add_wall_frame_left = self._add_wall_intervals[randint(0, len(self._add_wall_intervals) - 1)]
            self._add_wall()

    def _update_active_wall(self):
        if self._active_wall.right < self._bird.rect.left:
            self._active_wall = None
            try:
                index = self._wall_list.index(self._active_wall)
            except ValueError:
                # Means the active wall was removed
                index = 0
            while index < len(self._wall_list):
                if self._wall_list[index].right > self._bird.rect.left:
                    self._active_wall = self._wall_list[index]
                    return
                index += 1

    def _clean_sprites(self):
        if self._wall_list and self._wall_list[0].right < 0:
            # Remove top and bottom walls
            self._walls.remove(self._wall_list.pop(0))



