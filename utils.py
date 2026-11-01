from colorama import Fore, Style


class Console:
    COLOR_MAP = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "reset": Style.RESET_ALL
    }

    def __init__(self, *, decimals=3, length=50, verbose=True):
        self.decimals = decimals
        self.length = length
        self.verbose = verbose

    def log(self, first, second=None, color=None, endl=False):
        if not self.verbose:
            return

        # A failed write (e.g. a log file in a synchronized folder) must never
        # stop a campaign; the results themselves are saved separately.
        try:
            if isinstance(first, str):
                print(self.string(first, second, color=color, endl=endl),
                      end="\n" if not endl else "", flush=True)
            elif isinstance(first, dict):
                print(self.dictionary(first), end="", flush=True)
        except OSError:
            pass

    def dictionary(self, obj):
        output = ""
        for first, second in obj.items():
            output += self.string(f"{first}", f"{second}", endl=True)
        return output

    def string(self, first, second=None, color=None, endl=False):
        start_code = self.COLOR_MAP.get(color, "")
        end_code = Style.RESET_ALL if color in self.COLOR_MAP else ""

        if second is None:
            text = start_code + first.ljust(self.length, '.') + end_code
        else:
            text = start_code + first.ljust(self.length - 1, '.') + ': '
            if isinstance(second, float):
                text += f"{second:.{self.decimals}f}"
            else:
                text += f"{second}"
            text += end_code

        if endl:
            text += "\n"

        return text
