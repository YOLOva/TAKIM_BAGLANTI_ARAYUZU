

import os
import sys


def run():
    AutoLabellerUI()
if __name__ == '__main__':
    if __package__:
        from .interface.gui import AutoLabellerUI
    else:
        sys.path.append(os.path.dirname(__file__) + '/..')
        from interface.gui import AutoLabellerUI
    run()

