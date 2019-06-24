from random import uniform
from typing import List

import numpy as np
from pygame.time import Clock

from bird_game import BirdGame
from neural_network import FastNeuralNetwork


def run_game_headless(nn: FastNeuralNetwork):
    game = BirdGame(headless=True)

    while 1:
        if game.is_game_over():
            return game.score
        if game.score > 4000:
            return game.score

        output = nn.feedfordward(
            np.array([
                [game.distance_to_wall[0] / 800]
                , [game.distance_to_wall[1] / 600]
                , [game.bird_center[1] / 600]
                , [game.bird_speed[1]]
            ])
        )
        game.next_frame(output[0][0] > 0.5)


def run_game_real_speed(nn: FastNeuralNetwork):
    game = BirdGame()
    clock = Clock()
    while 1:
        if game.is_game_over():
            return game.score
        if game.score > 4000:
            return game.score

        output = nn.feedfordward(
            np.array([
                [game.distance_to_wall[0] / 800]
                , [game.distance_to_wall[1] / 600]
                , [game.bird_center[1] / 600]
                , [game.bird_speed[1]]
            ])
        )
        game.next_frame(output[0][0] > 0.5)
        clock.tick(200)


def run_generation(neural_networks: List[FastNeuralNetwork]):
    return [
        (run_game_headless(neural_network), neural_network)
        for neural_network in neural_networks
    ]


def create_random_layer(previous_layer_size, layer_size):
    return list([
        tuple(uniform(-1, 1) for _ in range(previous_layer_size))
        for _ in range(layer_size)
    ])


def get_first_neural_networks(nb):
    nns = []
    for _ in range(nb):
        nn = FastNeuralNetwork()
        nn.add_layer(np.array(create_random_layer(4, 10)))
        nn.add_layer(np.array(create_random_layer(10, 10)))
        nn.add_layer(np.array(create_random_layer(10, 1)))

        nns.append(nn)

    return nns


def mutate_nns(base_nns, nb_mutation):
    nns = []
    for base_nn in base_nns:
        nns.append(
            base_nn
        )
        for _ in range(nb_mutation - 1):
            nns.append(
                base_nn.mutate(10, 10)
            )

    return nns


def main():
    nb_per_iteration = 500
    nb_to_keep = 10
    nns = get_first_neural_networks(nb_per_iteration)

    for i in range(1, 101):
        print("Running generation {}".format(i))
        score_nns = run_generation(nns)
        score_nns.sort(key=lambda score_nn: score_nn[0], reverse=True)
        run_game_real_speed(score_nns[0][1])
        print("Mutating")
        nns = mutate_nns(
            [score_nn[1] for score_nn in score_nns[0:nb_to_keep]]
            , nb_per_iteration // nb_to_keep
        )
        print("Best score:", score_nns[0][0])
        print("Average 10 best scores:", sum([score_nn[0] for score_nn in score_nns[:10]]) / 10)
        print("Average score:", sum([score_nn[0] for score_nn in score_nns]) / len(score_nns))
        print()


if __name__ == '__main__':
    main()
