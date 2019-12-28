from environment import Grid2DSize, SnakesBackendSync
from agents import RandomPlayer, KeyboardPlayer, GreedyAgent, BetterGreedyAgent
from submission import MinimaxAgent, AlphaBetaAgent, TournamentAgent
from optparse import OptionParser
import sys
import csv


def start_demo_game(n_agents: int, game_duration: int, board_width: int, board_height: int,
                    n_fruits: int, use_keyboard_listener: bool):
    players = [KeyboardPlayer(use_keyboard_listener=use_keyboard_listener)] + [GreedyAgent() for _ in
                                                                               range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=not use_keyboard_listener)


def start_game_with_players(players, game_duration: int, board_width: int, board_height: int, n_fruits: int,
                            fast_run: bool = False, graphics_off: bool = False, depth: int = 5, length=None,
                            time: list = None):
    if length is None:
        length = [0]
    if len(players) < 1:
        print("The number of agents must be at least 1.")

    env = SnakesBackendSync(players,
                            grid_size=Grid2DSize(board_width, board_height),
                            n_fruits=n_fruits,
                            game_duration_in_turns=game_duration, depth=depth)
    env.run_game(human_speed=not fast_run, render=not graphics_off, length=length, delta_time=time)


def start_part_c(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):
    players = [BetterGreedyAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run,
                            graphics_off=graphics_off)


def start_part_d(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):
    players = [MinimaxAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run,
                            graphics_off=graphics_off)


def start_part_e(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):
    players = [AlphaBetaAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run,
                            graphics_off=graphics_off)


def start_part_g(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):

    length_1 = [0]
    time_1 = [0]
    for i in range(1):
        players = [GreedyAgent() for _ in range(n_agents)]
        start_game_with_players(players,
                                game_duration,
                                board_width,
                                board_height,
                                n_fruits,
                                fast_run=fast_run,
                                graphics_off=graphics_off,
                                length=length_1,
                                time=time_1)

    print(length_1[0]/10, time_1[0]/10)
    length_2 = [0]
    time_2 = [0]
    for i in range(1):
        players = [BetterGreedyAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
        start_game_with_players(players,
                                game_duration,
                                board_width,
                                board_height,
                                n_fruits,
                                fast_run=fast_run,
                                graphics_off=graphics_off,
                                length=length_2,
                                time=time_2)
    print(length_2[0] / 10, time_2[0] / 10)
    length_3 = [[0], [0], [0]]
    time_3 = [[0], [0], [0]]
    for depth in [2, 3, 4]:
        for i in range(1):
            print(depth)
            players = [MinimaxAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
            start_game_with_players(players,
                                    game_duration,
                                    board_width,
                                    board_height,
                                    n_fruits,
                                    fast_run=fast_run,
                                    graphics_off=graphics_off,
                                    depth=depth,
                                    length=length_3[int(depth - 2)],
                                    time=time_3[int(depth - 2)])

        print(length_3[int(depth - 2)][0] / 10, time_3[int(depth - 2)][0] / 10)
    length_4 = [[0], [0], [0]]
    time_4 = [[0], [0], [0]]
    for depth in [2, 3, 4]:
        for i in range(1):
            players = [AlphaBetaAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
            start_game_with_players(players,
                                    game_duration,
                                    board_width,
                                    board_height,
                                    n_fruits,
                                    fast_run=fast_run,
                                    graphics_off=graphics_off,
                                    depth=depth,
                                    length=length_4[int(depth - 2)],
                                    time=time_4[int(depth - 2)])
        # print(length_4[int(depth-2)][0] / 10, time_4[int(depth-2)][0] / 10)

    with open('experiment.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['GreedyAgent', length_1[0] / 10, time_1[0] / 10])
        writer.writerow(['betterAgent', length_2[0] / 10, time_2[0] / 10])
        for i in range(3):
            writer.writerow(['MinMaxAgent', i+2, length_3[i][0] / 10, time_3[i][0] / 10])
        for i in range(3):
            writer.writerow(['AlphaBetaAgent', i+2, length_4[i][0] / 10, time_4[i][0] / 10])


def start_custom_game(p1: str, p2: str, game_duration: int, board_width: int, board_height: int,
                      n_fruits: int, fast_run: bool, graphics_off: bool, use_keyboard_listener: bool):
    def get_player(p: str):
        if p == 'KeyboardPlayer':
            return KeyboardPlayer(use_keyboard_listener=use_keyboard_listener)
        elif p == 'GreedyAgent':
            return GreedyAgent()
        elif p == 'BetterGreedyAgent':
            return BetterGreedyAgent()
        elif p == 'MinimaxAgent':
            return MinimaxAgent()
        elif p == 'AlphaBetaAgent':
            return AlphaBetaAgent()
        elif p == 'TournamentAgent':
            return TournamentAgent()

    players = [get_player(p1), get_player(p2)]

    fast_run = True if use_keyboard_listener == False else fast_run
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run)


def get_user_command(argv):
    usage_str = """
    USAGE: python main.py <options>
    """
    parser = OptionParser(usage_str)

    parser.add_option('-d', '--demo', dest='demo_run',
                      help="Run a demo game, with interactive keyboard",
                      default=True)
    parser.add_option('-f', '--fast', dest='fast_run', default=False, action='store_true', help='Run Fast')
    parser.add_option('-g', '--graphics_off', dest='graphics_off', default=False, action='store_true')
    parser.add_option('--n_agents', dest='n_agents', type=int,
                      help="number of snakes in the game", default=2)
    parser.add_option('-N', '--game_duration', dest='game_duration', type=int,
                      help="Number of turns in a game", default=500)
    parser.add_option('-W', '--board-width', dest='board_width', type=int,
                      help="width of the board", default=50)
    parser.add_option('-H', '--board-height', dest='board_height', type=int,
                      help="height of the board", default=50)
    parser.add_option('--n_fruits', dest='n_fruits', type=int,
                      help="number of fruits on the board", default=51)
    parser.add_option('--kb_listener_off', dest='use_keyboard_listener', default=True, action='store_false',
                      help="Whether or not to use keyboard listener for logging KB events. If you use Mac OS you MUST use False!")
    parser.add_option('--part_c', dest='part_c', default=False, action='store_true',
                      help="whether to run part C.")
    parser.add_option('--part_d', dest='part_d', default=False, action='store_true',
                      help="whether to run part D.")
    parser.add_option('--part_e', dest='part_e', default=False, action='store_true',
                      help="whether to run part E.")
    parser.add_option('--part_g', dest='part_g', default=False, action='store_true')
    parser.add_option('--custom_game', dest='custom_game', default=False, action='store_true',
                      help="whether to run a custom game with user defined players.")
    parser.add_option('--p1', dest='player1', default='KeyboardPlayer',
                      help="The class name of player 1 in the custom game")
    parser.add_option('--p2', dest='player2', default='GreedyPlayer',
                      help="The class name of player 2 in the custom game")

    options, other = parser.parse_args(argv)
    if len(other):
        raise Exception(f'wrong usage: {other}')

    if options.part_c:
        start_part_c(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.part_d:
        start_part_d(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.part_e:
        start_part_e(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.part_g:
        start_part_g(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.custom_game:
        start_custom_game(options.player1,
                          options.player2,
                          options.game_duration,
                          options.board_width,
                          options.board_height,
                          options.n_fruits,
                          options.fast_run,
                          options.graphics_off,
                          options.use_keyboard_listener
                          )
    elif options.demo_run:
        start_demo_game(options.n_agents,
                        options.game_duration,
                        options.board_width,
                        options.board_height,
                        options.n_fruits,
                        options.use_keyboard_listener)


if __name__ == '__main__':
    get_user_command(sys.argv[1:])