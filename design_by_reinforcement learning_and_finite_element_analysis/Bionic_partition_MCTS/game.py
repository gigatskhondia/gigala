# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
from utils import match_lines_to_coordinates, checkpoints_visit,has_at_least_two_neighbors,\
weight_and_strength_did_not_deteriorate, draw


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 5))
        self.height = int(kwargs.get('height', 5))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
       # # self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        self.weight = float("inf")
        self.strength = -float("inf")
        # self.new_strength = kwargs.get("new_strength", -float("inf"))
        # self.new_weight = kwargs.get("new_weight", float("inf"))
        self.dx =kwargs.get('dx', 1)
        self.dy = kwargs.get('dy', 1)
        self.checkpoints=kwargs.get('checkpoints',[(0,0),(4,0),(0,4),(4,4)])
        self.win_limit = 40

    def init_board(self, start_player=0):
        ## if self.width < self.n_in_row or self.height < self.n_in_row:
        ##     raise Exception('board width and height can not be '
        ##                     'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range((self.width-1) * (self.height-1)*4+self.width+self.height-2))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):

        h = move // self.height
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range((self.width-1) * (self.height-1)*4+self.width+self.height-2):
            return -1
        return move

   # # def current_state(self):
    ##     """return the board state from the perspective of the current player.
    ##     state shape: 4*width*height
    ##     """
    ##
    ##     square_state = np.zeros((4, self.width, self.height))
    ##     if self.states:
    ##         moves, players = np.array(list(zip(*self.states.items())))
    ##         move_curr = moves[players == self.current_player]
    ##         move_oppo = moves[players != self.current_player]
    ##         square_state[0][move_curr // self.width,
    ##                         move_curr % self.height] = 1.0
    ##         square_state[1][move_oppo // self.width,
    ##                         move_oppo % self.height] = 1.0
    ##         # indicate the last move location
    ##         square_state[2][self.last_move // self.width,
    ##                         self.last_move % self.height] = 1.0
    ##     if len(self.states) % 2 == 0:
    ##         square_state[3][:, :] = 1.0  # indicate the colour to play
    ##     return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        ## n = self.n_in_row

        moved = list(set(range((width-1) * (height-1)*4+(width-1) + (height-1))) - set(self.availables))
        ## if len(moved) < self.n_in_row *2-1:
        ##     return False, -1
        self.moved=moved

        self.lines_dic=match_lines_to_coordinates([width,height],self.dx,self.dy)

        if checkpoints_visit(moved, self.checkpoints, self.lines_dic) == True and \
                has_at_least_two_neighbors(moved, self.lines_dic) == True:
            flag, new_weight, new_strength, coord, elcon = weight_and_strength_did_not_deteriorate(moved, self.weight, self.strength , self.checkpoints, self.lines_dic)
            player = states[self.last_move]

            # print(flag, new_weight,new_strength)

            self.coord = coord
            self.elcon = elcon

            if flag and len(moved) <= self.win_limit:
                self.weight=new_weight
                self.strength=new_strength
                return True, player
            else:
                return True, -1
            # else:
            #     if player==self.players[0]:
            #         return True,self.players[1]
            #     else:
            #         return True, self.players[0]

        ## for m in moved:
        ##     h = m // width
        ##     w = m % width
        ##     player = states[m]
        ##
        ##     if (w in range(width - n + 1) and
        ##             len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
        ##         return True, player
        ##
        ##     if (h in range(height - n + 1) and
        ##             len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
        ##         return True, player
        ##
        ##     if (w in range(width - n + 1) and h in range(height - n + 1) and
        ##             len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
        ##         return True, player
        ##
        ##     if (w in range(n - 1, width) and h in range(height - n + 1) and
        ##             len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
        ##         return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()

        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        t=1
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            print("move {}: ".format(t), move)
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()

            if end:
                print("winner: ", winner)

                _, new_weight_1, new_strength_1, coord_1, elcon_1 = weight_and_strength_did_not_deteriorate(self.board.moved,
                                                                                                       self.board.weight,
                                                                                                       self.board.strength,
                                                                                                       self.board.checkpoints,
                                                                                                       self.board.lines_dic)
                print("Structure's")
                print("weight: ", new_weight_1,  "strength: ", new_strength_1)
                print(draw("green", coord_1, elcon_1))
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner
            t+=1
            # new
            # self.board.strength = -float("inf")
            # self.board.weight = float("inf")

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # print(self.board.weight, self.board.strength)
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
