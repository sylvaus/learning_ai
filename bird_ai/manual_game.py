import time

import pygame
from pygame.time import Clock

from bird_game import BirdGame


def wait_for_click():
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return
            if event.type == pygame.QUIT:
                quit()

        time.sleep(0.5)


def main():
    game = BirdGame()
    clock = Clock()

    wait_for_click()
    while 1:
        key_up = False
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                key_up = True
            if event.type == pygame.QUIT:
                quit()

        print("Score: {}. Distance: {}".format(game.score, game.distance_to_wall))
        if game.is_game_over():
            wait_for_click()
            game.reset()

        game.next_frame(key_up)
        clock.tick(60)


if __name__ == '__main__':
    main()
