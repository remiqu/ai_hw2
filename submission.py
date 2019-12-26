from environment import Player, GameState, GameAction, get_next_state, time
from utils import get_fitness
import numpy as np
from enum import Enum


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    if not state.snakes[player_index].alive:
        return 0

    head = state.snakes[player_index].head
    sum = 100 * state.snakes[player_index].length

    for fruit in state.fruits_locations:
        sum += 1 / (abs(fruit[0] - head[0]) + abs(fruit[1] - head[1]))

    return sum


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    def __init__(self, depth=5):
        self.depth = depth

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def utility(self, state: TurnBasedGameState) -> float:
        return np.inf if state.game_state.current_winner == self.player_index else -np.inf

    def RB_minimax(self, state: TurnBasedGameState, depth):
        if state.game_state.is_terminal_state:
            return self.utility(state)
        if depth == 0:
            return heuristic(state.game_state, self.player_index)
        if state.turn == self.Turn.AGENT_TURN:
            cur_max = -np.inf
            for action in state.game_state.get_possible_actions(player_index=self.player_index):
                next_state = self.TurnBasedGameState(state.game_state, action)
                v = self.RB_minimax(next_state, depth)
                cur_max = max(v, cur_max)
            return cur_max
        else:
            cur_min = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                tb_next_state = self.TurnBasedGameState(next_state, None)
                v = self.RB_minimax(tb_next_state, depth - 1)
                cur_min = min(v, cur_min)
            return cur_min

    def get_action(self, state: GameState, delta_time=None) -> GameAction:
        start_time = time.time()
        best_value = -np.inf
        best_actions = state.get_possible_actions(player_index=self.player_index)
        for action in state.get_possible_actions(player_index=self.player_index):
            next_state = self.TurnBasedGameState(state, action)
            max_value = self.RB_minimax(next_state, state.depth)
            if max_value > best_value:
                best_value = max_value
                best_actions = [action]
            elif best_value == max_value:
                best_actions.append(action)
        end_time = time.time()
        delta_time[0] += end_time - start_time
        return np.random.choice(best_actions)

    # def max_value(self, state: TurnBasedGameState, depth) -> float:
    #     if state.game_state.is_terminal_state:
    #         return self.utility(state)
    #     if depth == 0:
    #         return heuristic(state.game_state, self.player_index)
    #     v = -np.inf
    #     for action in state.game_state.get_possible_actions(player_index=self.player_index):
    #         next_state = self.TurnBasedGameState(state.game_state, action)
    #         v = max(v, self.min_value(next_state, depth-1))
    #     return v
    #
    # def min_value(self, state: TurnBasedGameState, depth) -> float:
    #     if state.game_state.is_terminal_state:
    #         return self.utility(state)
    #     if depth == 0:
    #         return heuristic(state.game_state, self.player_index)
    #     v = np.inf
    #
    #     # todo: no need to pass to this function the id of the Player index that we are trying right now?
    #     for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
    #                                                                                       player_index=self.player_index):
    #         opponents_actions[self.player_index] = state.agent_action
    #         next_state = get_next_state(state.game_state, opponents_actions)
    #         tb_next_state = self.TurnBasedGameState(next_state, None)
    #         v = min(v, self.max_value(tb_next_state, depth-1))
    #     return v
    #
    # def get_action(self, state: GameState) -> GameAction:
    #     best_value = np.inf
    #     best_actions = state.get_possible_actions(player_index=self.player_index)
    #     for action in state.get_possible_actions(player_index=self.player_index):
    #         next_state = self.TurnBasedGameState(state, action)
    #         min_value = self.min_value(next_state, self.depth-1)
    #         if min_value < best_value:
    #             best_value = min_value
    #             best_actions = [action]
    #         elif best_value == min_value:
    #             best_actions.append(action)
    #     return np.random.choice(best_actions)

    # def utility(self, state: TurnBasedGameState) -> list:
    #     return [s.length for s in state.snakes if s.alive and -1 if not s.alive]
    #
    #
    #
    # def max_value(self, state: TurnBasedGameState) -> list:
    #     if state.game_state.is_terminal_state:
    #         return self.utility(state)
    #     v = -np.inf
    #     for action in state.game_state.get_possible_actions(player_index=self.player_index):
    #         next_state = self.TurnBasedGameState(state.game_state, action)
    #         v = max_utility(v, self.min_value(next_state))
    #     return v
    #
    # def min_value(self, state: TurnBasedGameState) -> list:
    #     if state.game_state.is_terminal_state:
    #         return self.utility(state)
    #     v = np.inf
    #     for opponents_actions in state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index):
    #         opponents_actions[self.player_index] = state.agent_action
    #         next_state = get_next_state(state.game_state, opponents_actions)
    #         tb_next_state = self.TurnBasedGameState(next_state, None)
    #         v = min_utility(v, self.max_value())
    #     return v


class AlphaBetaAgent(MinimaxAgent):

    def RB_alphaBeta(self, state: MinimaxAgent.TurnBasedGameState, depth, alpha, beta):
        if state.game_state.is_terminal_state:
            return self.utility(state)
        if depth == 0:
            return heuristic(state.game_state, self.player_index)
        if state.turn == self.Turn.AGENT_TURN:
            cur_max = -np.inf
            for action in state.game_state.get_possible_actions(player_index=self.player_index):
                next_state = self.TurnBasedGameState(state.game_state, action)
                v = self.RB_alphaBeta(next_state, depth - 1, alpha, beta)
                cur_max = max(v, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf
            return cur_max
        else:
            cur_min = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                tb_next_state = self.TurnBasedGameState(next_state, None)
                v = self.RB_alphaBeta(tb_next_state, depth, alpha, beta)
                cur_min = min(v, cur_min)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf
            return cur_min

    def get_action(self, state: GameState, delta_time=None) -> GameAction:
        start_time = time.time()
        best_value = -np.inf
        best_actions = state.get_possible_actions(player_index=self.player_index)
        for action in state.get_possible_actions(player_index=self.player_index):
            next_state = self.TurnBasedGameState(state, action)
            max_value = self.RB_alphaBeta(next_state, state.depth - 1, -np.inf, np.inf)
            if max_value > best_value:
                best_value = max_value
                best_actions = [action]
            elif best_value == max_value:
                best_actions.append(action)
        end_time = time.time()
        delta_time[0] += end_time - start_time
        return np.random.choice(best_actions)


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.
    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """



def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.
    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """



    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()