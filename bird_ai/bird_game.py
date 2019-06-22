from time import sleep

import pygame
from pygame import Vector2

from pygame.sprite import Sprite, Group
from pygame.time import Clock

WHITE = (255, 255, 255)


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
            self, center=None, speed=None, gravity=0.6, flap_accel=-13
            , max_abs_vel=None, *groups):
        super().__init__(*groups)
        self.image = pygame.image.load("images/bird.png")
        self.rect = self.image.get_rect()
        if center:
            self.rect.center = center

        self._pos = Vector2(*self.rect.center)
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
    def __init__(self, center=None, *groups):
        super().__init__(*groups)
        self.image = pygame.image.load("images/wall.png")
        self.rect = self.image.get_rect()
        if center:
            self.rect.center = center


class BirdGame:
    def __init__(self, size=(800, 600)):
        pygame.init()
        self._screen = pygame.display.set_mode(size)
        self._bird = Bird((size[0]//10, size[1]//2))
        self._walls = Group()
        self._all_sprites = Group()
        self._all_sprites.add(self._bird)
        self._game_over = False

    def is_game_over(self):
        return self._game_over

    def next_frame(self, key_up):
        if key_up:
            self._bird.flap()
        self._all_sprites.update()

        self._draw()

    def _draw(self):
        self._screen.fill(WHITE)
        self._all_sprites.draw(self._screen)

        pygame.display.flip()


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


def old():
    size = width, height = 800, 800
    speed = [2, 0]
    black = 0, 0, 0
    screen = pygame.display.set_mode(size)
    ball = pygame.image.load("images/wall.png")
    ballrect = ball.get_rect()
    ballrect = ballrect.move([0, -800])
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        ballrect = ballrect.move(speed)
        if ballrect.left < 0 or ballrect.right > width:
            speed[0] = -speed[0]
        if ballrect.top < 0 or ballrect.bottom > height:
            pass #speed[1] = -speed[1]

        sleep(0.01)
        screen.fill(black)
        screen.blit(ball, ballrect)
        pygame.display.flip()


if __name__ == '__main__':
    main()