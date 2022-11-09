import copy
from numba import njit
import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

INF = 999999

time_shoot = 1234567

order0 = [(6, 5), (4, 6), (2, 6), (4, 1), (1, 1), (3, 6), (6, 6), (1, 3), (1, 4), (6, 4), (1, 6), (6, 2),
          (1, 5), (5, 6),
          (6, 1), (5, 1), (6, 3), (1, 2), (3, 1), (2, 1),
          (5, 4), (3, 2), (4, 2), (2, 5), (2, 4), (2, 2), (4, 5), (5, 5), (5, 3), (3, 5), (2, 3), (5, 2),
          (3, 3), (4, 3), (3, 4), (4, 4),
          (1, 0), (1, 7), (0, 5), (4, 0), (7, 0), (0, 4), (0, 6), (2, 7), (3, 0), (7, 6), (7, 1), (0, 3),
          (6, 7), (7, 5),
          (7, 3), (5, 7), (0, 7), (7, 2), (7, 7), (3, 7), (5, 0), (6, 0), (7, 4), (0, 2), (0, 0), (2, 0),
          (4, 7), (0, 1)]

winner_order = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
    (7, 6), (7, 7),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
    (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
    (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
    (3, 2), (4, 2), (3, 5), (4, 5),
    (3, 3), (3, 4), (4, 3), (4, 4)]

Dir = [(-1, -1), (1, -1), (-1, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (0, 1)]

Dir2 = [(1, 0), (0, -1), (-1, 0), (0, 1)]


def calculate(stat: list, dp_array):
    stat_hash: int = 0
    pow3: int = 1
    for i in range(8):
        stat_hash += (stat[i] + 1) * pow3
        pow3 *= 3
    if dp_array[stat_hash] > -9999:
        return dp_array[stat_hash]
    zero = 0
    for _ in range(8):
        zero += 1 if stat[_] == COLOR_NONE else 0
    res = 0.0
    stat_0 = [0 for _ in range(8)]
    for j in range(8):
        stat_0[j] = stat[j]
    if zero == 0:
        res += (stat[0] + stat[7]) * 2
        for i in range(1, 7):
            res += stat[i]
    else:
        for i in range(8):
            if stat[i] == COLOR_NONE:
                flag = 0
                l_, r_ = i - 1, i + 1
                l_list, r_list = [], []
                if l_ >= 0 and stat[l_] != COLOR_NONE:
                    color_l = stat[l_]
                    while l_ >= 0 and stat[l_] == color_l:
                        l_ -= 1
                    if l_ >= 0 and l_ != i - 1 and stat[l_] == -color_l:
                        flag += color_l
                        for j in range(l_ + 1, i):
                            l_list.append(j)
                if r_ < 8 and stat[r_] != COLOR_NONE:
                    color_r = stat[r_]
                    while r_ < 8 and stat[r_] == color_r:
                        r_ += 1
                    if r_ < 8 and r_ != i + 1 and stat[r_] == -color_r:
                        flag += color_r
                        for j in range(i + 1, r_):
                            r_list.append(j)

                stat[i] = -1
                flag_l, flag_r = False, False
                if len(l_list) > 0 and stat[l_list[0]] == 1:
                    flag_l = True
                    for j in l_list:
                        stat[j] = -1
                if len(r_list) > 0 and stat[r_list[0]] == 1:
                    flag_r = True
                    for j in r_list:
                        stat[j] = -1
                res += calculate(stat, dp_array) * 1 / zero * (1 / 2 - flag * 1 / 6)
                if flag_l:
                    for j in l_list:
                        stat[j] = 1
                if flag_r:
                    for j in r_list:
                        stat[j] = 1

                flag_l, flag_r = False, False
                stat[i] = 1
                if len(l_list) > 0 and stat[l_list[0]] == -1:
                    flag_l = True
                    for j in l_list:
                        stat[j] = 1
                if len(r_list) > 0 and stat[r_list[0]] == -1:
                    flag_r = True
                    for j in r_list:
                        stat[j] = 1
                res += calculate(stat, dp_array) * 1 / zero * (1 / 2 + flag * 1 / 6)
                if flag_l:
                    for j in l_list:
                        stat[j] = -1
                if flag_r:
                    for j in r_list:
                        stat[j] = -1
                stat[i] = 0
    dp_array[stat_hash] = res
    return res


def bit_board(chessboard) -> tuple:
    a, b = 0, 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_BLACK:
                a |= (1 << (i * 8 + j))
            elif chessboard[i][j] == COLOR_WHITE:
                b |= (1 << (i * 8 + j))
    return a, b


def legal_(p):
    return (0 <= p[0] < 8) and (0 <= p[1] < 8)


def get_accessible_(chessboard: np.array, turn, pos) -> bool:
    if chessboard[pos[0]][pos[1]] != COLOR_NONE:
        return False
    for k in range(8):
        p = (pos[0] + Dir[k][0], pos[1] + Dir[k][1])
        while legal_(p) and chessboard[p[0]][p[1]] == -turn:
            p = (p[0] + Dir[k][0], p[1] + Dir[k][1])
        if legal_(p) and p != (pos[0] + Dir[k][0], pos[1] + Dir[k][1]) and chessboard[p[0]][
            p[1]] == turn:
            return True
    return False


def get_next_(chessboard: np.array, turn, pos):
    res, res_tmp = [], []
    for k in range(8):
        res_tmp.clear()
        p = (pos[0] + Dir[k][0], pos[1] + Dir[k][1])
        while legal_(p) and chessboard[p[0]][p[1]] == -turn:
            res_tmp.append(p)
            p = (p[0] + Dir[k][0], p[1] + Dir[k][1])
        if legal_(p) and p != (pos[0] + Dir[k][0], pos[1] + Dir[k][1]) and chessboard[p[0]][p[1]] == turn:
            res += res_tmp
    for pos0 in res:
        chessboard[pos0[0]][pos0[1]] = turn
    chessboard[pos[0]][pos[1]] = turn


def get_accessible_list_(chessboard: np.array, turn) -> list:
    """
    Get a list that contains all valid position.

    :param chessboard: the current chessboard
    :param turn: the color of the side to be dropped
    :return: the list
    """
    lis = []
    for i in range(8):
        for j in range(8):
            if get_accessible_(chessboard, turn, (i, j)):
                lis.append((i, j))
    return lis


@njit()
def calculate0(s0, s1, s2, s3, s4, s5, s6, s7, dp_array):
    x0: int = (s0 + 1) + (s1 + 1) * 3 + (s2 + 1) * 9 + (s3 + 1) * 27 + (s4 + 1) * 81 + (s5 + 1) * 243 + (
            s6 + 1) * 729 + (s7 + 1) * 2187
    return dp_array[x0]


@njit()
def num_chess(chessboard: np.array, turn) -> tuple:
    """
    number of chess in a chessboard.

    :param turn: who is playing
    :param chessboard: the current chessboard
    :return: an integer
    """
    sc = 0
    my = 0
    for i in range(8):
        for j in range(8):
            sc += (1 if chessboard[i][j] != COLOR_NONE else 0)
            my += (1 if chessboard[i][j] == turn else 0)
    return sc, my


@njit()
def legal(p):
    """
    To check whether position p is out of bounds.

    :param p: position
    :return: whether p is out-of-bound
    """
    return (0 <= p[0] < 8) and (0 <= p[1] < 8)

@njit()
def get_accessible(chessboard: np.array, now_turn, pos, Dir0: np.array) -> bool:
    """
    Check if the current move at pos is legal.

    :param Dir0:
    :param chessboard: the current chessboard
    :param now_turn: the color of the side that will play
    :param pos: the position to be dropped
    :return: whether pos is legal
    """
    if chessboard[pos[0]][pos[1]] != COLOR_NONE:
        return False
    for kk in Dir0:
        p = (pos[0] + kk[0], pos[1] + kk[1])
        while legal(p) and chessboard[p[0]][p[1]] == -now_turn:
            p = (p[0] + kk[0], p[1] + kk[1])
        if legal(p) and p != (pos[0] + kk[0], pos[1] + kk[1]) and chessboard[p[0]][p[1]] == now_turn:
            return True
    return False

@njit()
def next_board(chessboard: np.array, pos, now_turn, Dir0: np.array) -> list:
    """
    Return the next chessboard when a move is made
    We default that pos is legal. Otherwise, the function will have unpredictable behavior.

    :param Dir0:
    :param chessboard: the current chessboard(will be changed to the next chessboard)
    :param now_turn: the color of the side that will play
    :param pos: the position to be dropped
    :return:
    """
    res, res_tmp = [(-1, -1)], [(-1, -1)]
    res.clear()
    for kk in Dir0:
        res_tmp.clear()
        p = (pos[0] + kk[0], pos[1] + kk[1])
        while legal(p) and chessboard[p[0]][p[1]] == -now_turn:
            res_tmp.append(p)
            p = (p[0] + kk[0], p[1] + kk[1])
        if legal(p) and p != (pos[0] + kk[0], pos[1] + kk[1]) and chessboard[p[0]][p[1]] == now_turn:
            res += res_tmp
    for pos0 in res:
        chessboard[pos0[0]][pos0[1]] = now_turn
    chessboard[pos[0]][pos[1]] = now_turn
    return res

@njit()
def get_accessible_list(chessboard: np.array, now_turn, Dir0: np.array) -> list:
    """
    Get a list that contains all valid position.

    :param Dir0:
    :param chessboard: the current chessboard
    :param now_turn: the color of the side to be dropped
    :return: the list
    """
    lis = [(-1, -1)]
    lis.clear()
    for i in range(8):
        for j in range(8):
            if get_accessible(chessboard, now_turn, (i, j), Dir0):
                lis.append((i, j))
    return lis


@njit()
def stable_calc(chessboard: np.array, stable_list: np.array, self_color):
    res = [0, 0]
    if chessboard[0][0] != COLOR_NONE:
        color_idx = 0 if chessboard[0][0] == self_color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[i][j] == chessboard[0][0]:
                stable_list[i][j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    if chessboard[7][0] != COLOR_NONE:
        color_idx = 0 if chessboard[7][0] == self_color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[7 - i][j] == chessboard[7][0]:
                stable_list[7 - i][j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    if chessboard[0][7] != COLOR_NONE:
        color_idx = 0 if chessboard[0][7] == self_color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[i][7 - j] == chessboard[0][7]:
                stable_list[i][7 - j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    if chessboard[7][7] != COLOR_NONE:
        color_idx = 0 if chessboard[7][7] == self_color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[7 - i][7 - j] == chessboard[7][7]:
                stable_list[7 - i][7 - j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    return res


@njit()
def initial_calc(chessboard: np.array, score_matrix: np.array, score_matrix1: np.array, self_color, numC):
    initial = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_NONE:
                continue
            if numC <= 44:
                initial += score_matrix[i][j] if chessboard[i][j] != self_color else -score_matrix[i][j]
            else:
                initial += score_matrix1[i][j] if chessboard[i][j] != self_color else -score_matrix1[i][j]
    return initial


@njit()
def front_calc(chessboard: np.array, is_stable: np.array, self_color, Dir0: np.array):
    front = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] != COLOR_NONE and not is_stable[i][j]:
                for kk in Dir0:
                    a = i + kk[0]
                    b = j + kk[1]
                    if legal((a, b)) and chessboard[a][b] == COLOR_NONE:
                        front += 1 if chessboard[i][j] == self_color else -1
    return front


@njit()
def arb_calc(chessboard: np.array, self_color, arb_matrix: np.array, arb_matrix1: np.array, sc, dir0: np.array):
    arb = 0
    arb_my = get_accessible_list(chessboard, self_color, dir0)
    arb_opp = get_accessible_list(chessboard, -self_color, dir0)
    for action in arb_my:
        arb += arb_matrix[action[0]][action[1]] if sc <= 44 else arb_matrix1[action[0]][action[1]]
    for action in arb_opp:
        arb -= arb_matrix[action[0]][action[1]] if sc <= 44 else arb_matrix1[action[0]][action[1]]
    return arb

@njit()
def winner(chessboard: np.array, turn: int, deep: int):
    sc, my = num_chess(chessboard, turn if (deep & 1) == 0 else -turn)
    return INF if (my < sc - my) else (0 if (my == sc - my) else -INF), -1, -1

@njit()
def final_search(dir0: np.array, chessboard: np.array, turn, deep: int, last_chance) -> tuple:
    if deep == 0:
        return time_shoot, -1, -1
    result = -INF if (deep & 1) == 0 else INF
    any_pos_flag = False
    final_pos_x, final_pos_y = -1, -1
    for i in range(8):
        for j in range(8):
            ite = (i, j)
            if get_accessible(chessboard, turn, ite, dir0):
                any_pos_flag = True
                # movement
                changed = next_board(chessboard, ite, turn, dir0)
                nxt = final_search(dir0, chessboard, -turn, deep - 1, True)
                chessboard[ite[0]][ite[1]] = COLOR_NONE
                for _ in range(len(changed)):
                    xx, yy = changed[_][0], changed[_][1]
                    chessboard[xx][yy] = -turn
                if nxt[0] > 1234560:
                    if (deep & 1) == 0:
                        return result, -1, -1
                    else:
                        return -INF, -1, -1
                if (deep & 1) == 0:
                    if result < nxt[0]:
                        result = nxt[0]
                        final_pos_x, final_pos_y = ite
                        if result == INF:
                            if deep == 100:
                                break
                            return result, -1, -1
                else:
                    if result > nxt[0]:
                        result = nxt[0]
                    if result == -INF:
                        return result, -1, -1
    # At the first layer, returns the position
    if deep == 100:
        return result, final_pos_x, final_pos_y
    # There is no location to go down (at this time to ensure that it will not appear in the first layer)
    if not any_pos_flag:
        sc, my = num_chess(chessboard, turn if (deep & 1) == 0 else -turn)
        if sc == 64 or not last_chance:
            return INF if (my < sc - my) else (0 if (my == sc - my) else -INF), -1, -1
        else:
            return final_search(dir0, chessboard, -turn, deep - 1, False)
    return result, final_pos_x, final_pos_y


class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        timerr = time.time()
        self.chessboard_size = chessboard_size

        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list.
        # The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.maxDepth = 1
        self.single_time_out = 0
        self.final_time_out = 0
        self.is_first_round = 3
        self.final_search_begin = 52

        self.score_matrix = np.array(
            [[-2, -8, 13, 15, 15, 13, -8, -2], [-8, 32, 59, 16, 16, 59, 32, -8], [13, 59, 52, -23, -23, 52, 59, 13],
             [15, 16, -23, -63, -63, -23, 16, 15], [15, 16, -23, -63, -63, -23, 16, 15],
             [13, 59, 52, -23, -23, 52, 59, 13],
             [-8, 32, 59, 16, 16, 59, 32, -8], [-2, -8, 13, 15, 15, 13, -8, -2]]
        )
        self.score_matrix1 = np.array(
            [[-18, -3, 10, -38, -38, 10, -3, -18], [-3, 40, 17, -18, -18, 17, 40, -3], [10, 17, 9, 55, 55, 9, 17, 10],
             [-38, -18, 55, 28, 28, 55, -18, -38], [-38, -18, 55, 28, 28, 55, -18, -38], [10, 17, 9, 55, 55, 9, 17, 10],
             [-3, 40, 17, -18, -18, 17, 40, -3], [-18, -3, 10, -38, -38, 10, -3, -18]]
        )
        self.arb_matrix = np.array(
            [[29, 52, 35, 4, 4, 35, 52, 29], [52, 42, 10, 35, 35, 10, 42, 52], [35, 10, 15, 59, 59, 15, 10, 35],
             [4, 35, 59, -47, -47, 59, 35, 4], [4, 35, 59, -47, -47, 59, 35, 4], [35, 10, 15, 59, 59, 15, 10, 35],
             [52, 42, 10, 35, 35, 10, 42, 52], [29, 52, 35, 4, 4, 35, 52, 29]]
        )
        self.arb_matrix1 = np.array(
            [[-4, 43, 60, -7, -7, 60, 43, -4], [43, -35, 18, -26, -26, 18, -35, 43],
             [60, 18, -34, -21, -21, -34, 18, 60],
             [-7, -26, -21, 23, 23, -21, -26, -7], [-7, -26, -21, 23, 23, -21, -26, -7],
             [60, 18, -34, -21, -21, -34, 18, 60], [43, -35, 18, -26, -26, 18, -35, 43],
             [-4, 43, 60, -7, -7, 60, 43, -4]]
        )
        self.c1, self.c2, self.c3, self.c4, self.c5 = 1, [245, 239, 144], [-52, -57, -57], [-4, -51, -92], [196, 85, 15]

        self.order_x = [(6, 5), (4, 6), (2, 6), (4, 1), (1, 1), (3, 6), (6, 6), (1, 3), (1, 4), (6, 4), (1, 6), (6, 2),
                        (1, 5), (5, 6),
                        (6, 1), (5, 1), (6, 3), (1, 2), (3, 1), (2, 1),
                        (5, 4), (3, 2), (4, 2), (2, 5), (2, 4), (2, 2), (4, 5), (5, 5), (5, 3), (3, 5), (2, 3), (5, 2),
                        (3, 3), (4, 3), (3, 4), (4, 4),
                        (1, 0), (1, 7), (0, 5), (4, 0), (7, 0), (0, 4), (0, 6), (2, 7), (3, 0), (7, 6), (7, 1), (0, 3),
                        (6, 7), (7, 5),
                        (7, 3), (5, 7), (0, 7), (7, 2), (7, 7), (3, 7), (5, 0), (6, 0), (7, 4), (0, 2), (0, 0), (2, 0),
                        (4, 7), (0, 1)]
        self.dir = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (0, 1)])
        self.dp_array = np.array([-9999.999 for _ in range(6561)])  # 3^8
        self.winner_order = np.array([
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),
            (7, 5),
            (7, 6), (7, 7),
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
            (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
            (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
            (3, 2), (4, 2), (3, 5), (4, 5),
            (3, 3), (3, 4), (4, 3), (4, 4)])
        self.kjk = {(34628173824, 68853694464): (2, 3), (34762915840, 68719476736): (4, 2),
                    (34829500416, 68719476736): (2, 4), (240786604032, 134217728): (5, 3),
                    (17695533694976, 134217728): (3, 5)}
        self.current_num = 0

    def evaluation(self, chessboard: np.array):
        """
        Evaluate a static situation.

        :param chessboard: the current chessboard
        :return: the evaluation value
        """
        # res0 = self.match.get(board_hash(chessboard, current_pos))
        # if res0 is not None:
        #     return res0
        is_stable = np.zeros((8, 8), dtype=bool)
        sc, my = num_chess(chessboard, self.color)
        stable_my, stable_opp = stable_calc(chessboard, is_stable, self.color)
        stable = (stable_opp - stable_my) * (1 + 0.2 * max(stable_opp, stable_my))
        diff = my + my - sc
        k = 0 if sc <= 24 else (1 if sc <= 44 else 2)
        initial = initial_calc(chessboard, self.score_matrix, self.score_matrix1, self.color, sc)
        arb = arb_calc(chessboard, self.color, self.arb_matrix, self.arb_matrix1, sc, self.dir)
        front = front_calc(chessboard, is_stable, self.color, self.dir)
        kkk = -1 if sc < 20 else (0 if sc <= 34 else (1 if sc <= 49 else 2))
        edge = 0
        if kkk >= 0:
            edge = calculate0(chessboard[0][0], chessboard[0][1], chessboard[0][2], chessboard[0][3], chessboard[0][4],
                              chessboard[0][5], chessboard[0][6], chessboard[0][7], self.dp_array)
            edge += calculate0(chessboard[7][0], chessboard[7][1], chessboard[7][2], chessboard[7][3], chessboard[7][4],
                               chessboard[7][5], chessboard[7][6], chessboard[7][7], self.dp_array)
            edge += calculate0(chessboard[0][0], chessboard[1][0], chessboard[2][0], chessboard[3][0], chessboard[4][0],
                               chessboard[5][0], chessboard[6][0], chessboard[7][0], self.dp_array)
            edge += calculate0(chessboard[0][7], chessboard[1][7], chessboard[2][7], chessboard[3][7], chessboard[4][7],
                               chessboard[5][7], chessboard[6][7], chessboard[7][7], self.dp_array)
            edge = 2 * self.c5[kkk] * edge
            if self.color == COLOR_WHITE:
                edge = -edge
        result = initial + self.c2[k] * stable + self.c3[k] * diff + self.c4[k] * front + edge + arb
        return result

    def ab(self, chessboard: np.array, turn, deep, alpha, beta, last_chance, timeout) -> tuple:
        if self.current_num + self.maxDepth - deep >= 56 and turn == self.color:
            x0 = final_search(self.dir, chessboard, turn, int(100), True)
            return x0[0], ((x0[1], x0[2]),)
        if deep <= 0:
            return self.evaluation(chessboard), ()
        if timeout > self.single_time_out:
            return time_shoot, ()
        start_time = time.time()
        any_pos_flag = 0
        final_list = ()
        for ite in self.order_x:
            if get_accessible(chessboard, turn, ite, self.dir):
                any_pos_flag += 1
                # movement
                changed = next_board(chessboard, ite, turn, self.dir)
                nxt = self.ab(chessboard, -turn, deep - 1, alpha, beta, True,
                              timeout + time.time() - start_time)
                chessboard[ite[0]][ite[1]] = COLOR_NONE
                for _ in range(len(changed)):
                    xx, yy = changed[_][0], changed[_][1]
                    chessboard[xx][yy] = -turn
                if nxt[0] > 1234560:
                    return time_shoot, final_list
                if turn == self.color:
                    if alpha < nxt[0]:
                        alpha = nxt[0]
                        final_list = (ite,) + nxt[1]
                    if alpha >= beta:
                        if self.maxDepth == deep:
                            break
                        return alpha, final_list
                else:
                    if beta > nxt[0]:
                        beta = nxt[0]
                        final_list = (ite,) + nxt[1]
                    if alpha >= beta:
                        return beta, final_list
        # At the first layer, returns the position
        if deep == self.maxDepth:
            return alpha, final_list
        # There is no location to go down (at this time to ensure that it will not appear in the first layer)
        elif any_pos_flag == 0:
            sc, my = num_chess(chessboard, self.color)
            if sc == 64 or not last_chance:
                return INF if (my < sc - my) else (0 if (my == sc - my) else -INF), ()
            else:
                return self.ab(chessboard, -turn, deep - 1, alpha, beta, False,
                               timeout + time.time() - start_time)
        return (alpha if turn == self.color else beta), final_list

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.candidate_list = get_accessible_list_(chessboard, self.color)
        self.current_num = num_chess(chessboard, 0)[0]

        if self.is_first_round == 3:
            kjk_pos = self.kjk.get(bit_board(chessboard))
            self.is_first_round = 2
            if kjk_pos is not None:
                self.candidate_list.append(kjk_pos)
            tmp_board = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -1, 0, 0, 0],
                [0, 0, 0, -1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0], ])
            winner(tmp_board, self.color, 0)
            get_accessible_list(tmp_board, self.color, self.dir)
            get_accessible(tmp_board, self.color, (0, 0), self.dir)
            next_board(tmp_board, (0, 0), self.color, self.dir)
            num_chess(tmp_board, self.color)
            calculate([0, 0, 0, 0, 0, 0, 0, 0], self.dp_array)
            calculate0(0, 0, 0, 0, 0, 0, 0, 0, self.dp_array)
            return None
        elif self.is_first_round == 2:
            timer_0 = time.time()
            self.evaluation(chessboard)
            self.single_time_out = 4.8 - (time.time() - timer_0)
            self.is_first_round = 1
        elif self.is_first_round == 1:
            timer_0 = time.time()
            self.is_first_round = 0
            final_search(self.dir, chessboard, self.color, 0, True)
            print("eva:", (time.time() - timer_0))
            self.single_time_out = 4.8 - (time.time() - timer_0)
        elif self.is_first_round == 0:
            self.single_time_out = 4.8

        if len(self.candidate_list) > 0:
            total_time = 0.0
            depth = 3
            best_list = ()
            best_list_set = set()

            while total_time < self.single_time_out:
                self.order_x.clear()
                best_list_set.clear()
                for i in range(len(best_list)):
                    if best_list[i] != (-1, -1):
                        self.order_x.append(best_list[i])
                        best_list_set.add(best_list[i])
                for i in order0:
                    if i not in best_list_set:
                        self.order_x.append(i)
                self.maxDepth = depth
                timer_ = time.time()
                x0 = self.ab(chessboard, self.color, self.maxDepth, -(INF + 1), INF + 1, True, total_time)
                best_list = x0[1]
                print("depth=", depth, "x0=", x0)
                total_time += time.time() - timer_
                if len(x0[1]) > 0 and x0[1][0] != (-1, -1):
                    self.candidate_list.append(x0[1][0])
                depth += 1
                if depth > 20:
                    break

            # if self.current_num >= self.final_search_begin:
            #     print("get winner search!")
            #     x1 = final_search(self.winner_order, self.dir, chessboard, self.color, int(100), True)
            #     print(x1[0])
            #     if x1[0] >= 0:
            #         print("get win or draw!")
            #         self.candidate_list.append((x1[1], x1[2]))
            #         print("search time =", total_time, "win step:", (x1[1], x1[2]))

    # print("time =", time.time() - timer)
    # ==============Find new pos========================================
    # Make sure that the position of your decision on the chess board is empty.
    # If not, the system will return error.
    # Add your decision into candidate_list, Records the chessboard
    # You need to add all the positions which are valid
    # candidate_list example: [(3,3),(4,4)]


class old_AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size

        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list.
        # The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.maxDepth = 1
        self.single_time_out = 0
        self.final_time_out = 0
        self.is_first_round = 3
        self.final_search_begin = 52

        self.score_matrix = np.array(
            [[58, -18, 20, -21, -21, 20, -18, 58],
             [-18, 59, 37, 44, 44, 37, 59, -18],
             [20, 37, 19, -31, -31, 19, 37, 20],
             [-21, 44, -31, -62, -62, -31, 44, -21],
             [-21, 44, -31, -62, -62, -31, 44, -21],
             [20, 37, 19, -31, -31, 19, 37, 20],
             [-18, 59, 37, 44, 44, 37, 59, -18],
             [58, -18, 20, -21, -21, 20, -18, 58]]
        )
        self.score_matrix1 = np.array(
            [[58, 31, -47, -53, -53, -47, 31, 58], [31, 58, 56, 61, 61, 56, 58, 31], [-47, 56, 32, 34, 34, 32, 56, -47],
             [-53, 61, 34, 33, 33, 34, 61, -53], [-53, 61, 34, 33, 33, 34, 61, -53], [-47, 56, 32, 34, 34, 32, 56, -47],
             [31, 58, 56, 61, 61, 56, 58, 31], [58, 31, -47, -53, -53, -47, 31, 58]]

        )
        self.arb_matrix = np.array(
            [[-6, 9, 9, 19, 19, 9, 9, -6], [9, 51, -16, -2, -2, -16, 51, 9], [9, -16, 25, 44, 44, 25, -16, 9],
             [19, -2, 44, -8, -8, 44, -2, 19], [19, -2, 44, -8, -8, 44, -2, 19], [9, -16, 25, 44, 44, 25, -16, 9],
             [9, 51, -16, -2, -2, -16, 51, 9], [-6, 9, 9, 19, 19, 9, 9, -6]]
        )
        self.arb_matrix1 = np.array(
            [[48, 2, 45, -48, -48, 45, 2, 48], [2, 42, 30, -37, -37, 30, 42, 2], [45, 30, 55, -50, -50, 55, 30, 45],
             [-48, -37, -50, 25, 25, -50, -37, -48], [-48, -37, -50, 25, 25, -50, -37, -48],
             [45, 30, 55, -50, -50, 55, 30, 45], [2, 42, 30, -37, -37, 30, 42, 2], [48, 2, 45, -48, -48, 45, 2, 48]]
        )
        self.c1, self.c2, self.c3, self.c4, self.c5 = 1, [3, 157, 62], [-11, -63, 3], [-12, -52, -119], [141, 47, 229]

        self.order_x = [(6, 5), (4, 6), (2, 6), (4, 1), (1, 1), (3, 6), (6, 6), (1, 3), (1, 4), (6, 4), (1, 6), (6, 2),
                        (1, 5), (5, 6),
                        (6, 1), (5, 1), (6, 3), (1, 2), (3, 1), (2, 1),
                        (5, 4), (3, 2), (4, 2), (2, 5), (2, 4), (2, 2), (4, 5), (5, 5), (5, 3), (3, 5), (2, 3), (5, 2),
                        (3, 3), (4, 3), (3, 4), (4, 4),
                        (1, 0), (1, 7), (0, 5), (4, 0), (7, 0), (0, 4), (0, 6), (2, 7), (3, 0), (7, 6), (7, 1), (0, 3),
                        (6, 7), (7, 5),
                        (7, 3), (5, 7), (0, 7), (7, 2), (7, 7), (3, 7), (5, 0), (6, 0), (7, 4), (0, 2), (0, 0), (2, 0),
                        (4, 7), (0, 1)]
        self.dir = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (0, 1)])
        self.dp_array = np.array([-9999.999 for _ in range(6561)])  # 3^8
        self.winner_order = np.array([
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),
            (7, 5),
            (7, 6), (7, 7),
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
            (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
            (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
            (3, 2), (4, 2), (3, 5), (4, 5),
            (3, 3), (3, 4), (4, 3), (4, 4)])
        self.kjk = {(34628173824, 68853694464): (2, 3), (34762915840, 68719476736): (4, 2),
                    (34829500416, 68719476736): (2, 4), (240786604032, 134217728): (5, 3),
                    (17695533694976, 134217728): (3, 5)}
        self.current_num = 0

    def evaluation(self, chessboard: np.array):
        """
        Evaluate a static situation.

        :param chessboard: the current chessboard
        :return: the evaluation value
        """
        # res0 = self.match.get(board_hash(chessboard, current_pos))
        # if res0 is not None:
        #     return res0
        is_stable = np.zeros((8, 8), dtype=bool)
        sc, my = num_chess(chessboard, self.color)
        stable_my, stable_opp = stable_calc(chessboard, is_stable, self.color)
        stable = (stable_opp - stable_my) * (1 + 0.2 * max(stable_opp, stable_my))
        diff = my + my - sc
        k = (sc - 5) // 20
        initial = initial_calc(chessboard, self.score_matrix, self.score_matrix1, self.color, sc)
        arb = arb_calc(chessboard, self.color, self.arb_matrix, self.arb_matrix1, sc, self.dir)
        front = front_calc(chessboard, is_stable, self.color, self.dir)
        kkk = (sc - 20) // 15
        edge = 0
        if kkk >= 0:
            edge = calculate0(chessboard[0][0], chessboard[0][1], chessboard[0][2], chessboard[0][3], chessboard[0][4],
                              chessboard[0][5], chessboard[0][6], chessboard[0][7], self.dp_array)
            edge += calculate0(chessboard[7][0], chessboard[7][1], chessboard[7][2], chessboard[7][3], chessboard[7][4],
                               chessboard[7][5], chessboard[7][6], chessboard[7][7], self.dp_array)
            edge += calculate0(chessboard[0][0], chessboard[1][0], chessboard[2][0], chessboard[3][0], chessboard[4][0],
                               chessboard[5][0], chessboard[6][0], chessboard[7][0], self.dp_array)
            edge += calculate0(chessboard[0][7], chessboard[1][7], chessboard[2][7], chessboard[3][7], chessboard[4][7],
                               chessboard[5][7], chessboard[6][7], chessboard[7][7], self.dp_array)
            edge = 2 * self.c5[kkk] * edge
            if self.color == COLOR_WHITE:
                edge = -edge
        result = self.c1 * initial + self.c2[k] * stable + self.c3[k] * diff + self.c4[k] * front + edge + arb
        return result

    def ab(self, chessboard: np.array, turn, deep, alpha, beta, last_chance, timeout) -> tuple:
        if self.current_num + self.maxDepth - deep >= 56 and turn == self.color:
            x0 = final_search(self.dir, chessboard, turn, int(100), True)
            return x0[0], ((x0[1], x0[2]),)
        if deep <= 0:
            return self.evaluation(chessboard), ()
        if timeout > self.single_time_out:
            return time_shoot, ()
        start_time = time.time()
        any_pos_flag = 0
        final_list = ()
        for ite in self.order_x:
            if get_accessible(chessboard, turn, ite, self.dir):
                any_pos_flag += 1
                # movement
                changed = next_board(chessboard, ite, turn, self.dir)
                nxt = self.ab(chessboard, -turn, deep - 1, alpha, beta, True,
                              timeout + time.time() - start_time)
                chessboard[ite[0]][ite[1]] = COLOR_NONE
                for _ in range(len(changed)):
                    xx, yy = changed[_][0], changed[_][1]
                    chessboard[xx][yy] = -turn
                if nxt[0] > 1234560:
                    return time_shoot, final_list
                if turn == self.color:
                    if alpha < nxt[0]:
                        alpha = nxt[0]
                        final_list = (ite,) + nxt[1]
                    if alpha >= beta:
                        if self.maxDepth == deep:
                            break
                        return alpha, final_list
                else:
                    if beta > nxt[0]:
                        beta = nxt[0]
                        final_list = (ite,) + nxt[1]
                    if alpha >= beta:
                        return beta, final_list
        # At the first layer, returns the position
        if deep == self.maxDepth:
            return alpha, final_list
        # There is no location to go down (at this time to ensure that it will not appear in the first layer)
        elif any_pos_flag == 0:
            sc, my = num_chess(chessboard, self.color)
            if sc == 64 or not last_chance:
                return INF if my < sc // 2 else (0 if my == sc // 2 else -INF), ()
            else:
                return self.ab(chessboard, -turn, deep - 1, alpha, beta, False,
                               timeout + time.time() - start_time)
        return (alpha if turn == self.color else beta), final_list

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.candidate_list = get_accessible_list_(chessboard, self.color)
        self.current_num = num_chess(chessboard, 0)[0]

        if self.is_first_round == 3:
            kjk_pos = self.kjk.get(bit_board(chessboard))
            self.is_first_round = 2
            if kjk_pos is not None:
                self.candidate_list.append(kjk_pos)
            tmp_board = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -1, 0, 0, 0],
                [0, 0, 0, -1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0], ])
            winner(tmp_board, self.color, 0)
            get_accessible_list(tmp_board, self.color, self.dir)
            get_accessible(tmp_board, self.color, (0, 0), self.dir)
            next_board(tmp_board, (0, 0), self.color, self.dir)
            num_chess(tmp_board, self.color)
            return None
        elif self.is_first_round == 2:
            timer_0 = time.time()
            self.evaluation(chessboard)
            calculate([0, 0, 0, 0, 0, 0, 0, 0], self.dp_array)
            calculate0(0, 0, 0, 0, 0, 0, 0, 0, self.dp_array)
            print("evaluation time =", time.time() - timer_0)
            self.single_time_out = 4.8 - (time.time() - timer_0)
            self.is_first_round = 1
        elif self.is_first_round == 1:
            timer_0 = time.time()
            self.is_first_round = 0
            final_search(self.dir, chessboard, self.color, 0, True)
            print("final initial time =", time.time() - timer_0)
            self.single_time_out = 4.8 - (time.time() - timer_0)
        elif self.is_first_round == 0:
            self.single_time_out = 4.8

        if len(self.candidate_list) > 0:
            total_time = 0.0
            depth = 3
            best_list = ()
            best_list_set = set()

            while total_time < self.single_time_out:
                self.order_x.clear()
                best_list_set.clear()
                for i in range(len(best_list)):
                    if best_list[i] != (-1, -1):
                        self.order_x.append(best_list[i])
                        best_list_set.add(best_list[i])
                for i in order0:
                    if i not in best_list_set:
                        self.order_x.append(i)
                self.maxDepth = depth
                timer_ = time.time()
                x0 = self.ab(chessboard, self.color, self.maxDepth, -(INF + 1), INF + 1, True, total_time)
                best_list = x0[1]
                print("depth=", depth, "x0=", x0)
                total_time += time.time() - timer_
                if len(x0[1]) > 0 and x0[1][0] != (-1, -1):
                    self.candidate_list.append(x0[1][0])
                depth += 1
                if depth > 20:
                    break

            # if self.current_num >= self.final_search_begin:
            #     print("get winner search!")
            #     x1 = final_search(self.winner_order, self.dir, chessboard, self.color, int(100), True)
            #     print(x1[0])
            #     if x1[0] >= 0:
            #         print("get win or draw!")
            #         self.candidate_list.append((x1[1], x1[2]))
            #         print("search time =", total_time, "win step:", (x1[1], x1[2]))

    # print("time =", time.time() - timer)
    # ==============Find new pos========================================
    # Make sure that the position of your decision on the chess board is empty.
    # If not, the system will return error.
    # Add your decision into candidate_list, Records the chessboard
    # You need to add all the positions which are valid
    # candidate_list example: [(3,3),(4,4)]


def board_print(chessboard: np.array):
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_BLACK:
                print("X", sep='', end='\t')
            elif chessboard[i][j] == COLOR_WHITE:
                print("O", sep='', end='\t')
            else:
                print("_", sep='', end='\t')
        print()


if __name__ == "__main__":
    random.seed()
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0], ])
    MACHINE_TURN = COLOR_BLACK
    now_turn0 = COLOR_BLACK
    kami = AI(8, MACHINE_TURN, 180)
    old_kami = old_AI(8, -MACHINE_TURN, 180)
    candidate_list_black = get_accessible_list_(board, COLOR_BLACK)
    candidate_list_white = get_accessible_list_(board, COLOR_WHITE)
    count = 4
    while len(candidate_list_black) > 0 or len(candidate_list_white) > 0:
        print("count =", count, "turn =", "BLACK" if now_turn0 == COLOR_BLACK else "WHITE",
              "AI " if now_turn0 == MACHINE_TURN else "OLD")
        timer = time.time()
        x = (-1, -1)
        if now_turn0 != MACHINE_TURN:
            if now_turn0 == COLOR_BLACK:
                old_kami.go(board)
                if len(candidate_list_black) > 0:
                    x = old_kami.candidate_list[-1]
                    get_next_(board, COLOR_BLACK, x)
            else:
                old_kami.go(board)
                if len(candidate_list_white) > 0:
                    x = old_kami.candidate_list[-1]
                    get_next_(board, COLOR_WHITE, x)
        else:
            if now_turn0 == COLOR_BLACK:
                kami.go(board)
                if len(candidate_list_black) > 0:
                    x = kami.candidate_list[-1]
                    get_next_(board, COLOR_BLACK, x)
            else:
                kami.go(board)
                if len(candidate_list_white) > 0:
                    x = kami.candidate_list[-1]
                    get_next_(board, COLOR_WHITE, x)
        print("TIME =", time.time() - timer)
        candidate_list_black = copy.deepcopy(get_accessible_list_(board, COLOR_BLACK))
        candidate_list_white = copy.deepcopy(get_accessible_list_(board, COLOR_WHITE))
        if not ((now_turn0 == COLOR_BLACK and len(candidate_list_white) == 0) or (
                now_turn0 == COLOR_WHITE and len(candidate_list_black) == 0)):
            now_turn0 = -now_turn0
        count += 1
        board_print(board)
        print("-------------------------")
    sc0, my0 = num_chess(board, MACHINE_TURN)
    print("final kami's num =", my0)
