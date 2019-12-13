from environment import Player, GameState, GameAction, get_next_state
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
    # Insert your code here...
    pass


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

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

    def utility(self, state: TurnBasedGameState) -> list:
        return 1 if state.game_state.current_winner == self.player_index else -1

    def max_value(self, state: TurnBasedGameState) -> list:
        if state.game_state.is_terminal_state:
            return self.utility(state)
        v = -np.inf
        for action in state.game_state.get_possible_actions(player_index=self.player_index):
            next_state = self.TurnBasedGameState(state.game_state, action)
            v = max(v, self.min_value(next_state))
        return v

    def min_value(self, state: TurnBasedGameState) -> list:
        if state.game_state.is_terminal_state:
            return self.utility(state)
        v = np.inf
        for opponents_actions in state.get_possible_actions_dicts_given_action(state.agent_action,player_index=self.player_index):
            opponents_actions[self.player_index] = state.agent_action
            next_state = get_next_state(state.game_state, opponents_actions)
            tb_next_state = self.TurnBasedGameState(next_state, None)
            v = min(v, self.max_value(tb_next_state))
        return v

    def get_action(self, state: GameState) -> GameAction:
        best_value = -np.inf
        best_actions = state.get_possible_actions(player_index=self.player_index)
        for action in state.game_state.get_possible_actions(player_index=self.player_index):
            next_state = self.TurnBasedGameState(state.game_state, action)
            if best_value > self.min_value(next_state):
                best_value = self.min_value(next_state)
                best_actions = [action]
            elif best_value == self.min_value(next_state):
                best_actions.append(action)
        return np.random.choice(best_actions)


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
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        pass


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
    pass


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

