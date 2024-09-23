import sys
import time
import numpy as np

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich.text import Text
from rich.console import Console, Group
from rich.progress import BarColumn

#from rich import print


def get_visualizers() -> dict:
    return {
        'bar': BarVisualizer,
        'list': ListVisualizer,
        'bubble': BubbleVisualizer,
        "none": NullVisualizer,
        "null": NullVisualizer,
    }

class Visualizer:
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    def display(self, outputs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class NullVisualizer(Visualizer):
    def display(self, outputs):
        pass

class RichVisualizer(Visualizer):
    def __init__(self, num_outputs, max_height=None):
        super().__init__(num_outputs)
        self.history = []
        self.console = Console()
        self.max_height = max_height or self.get_console_height()
        self.live = Live(console=self.console, refresh_per_second=10)
        self.bubble_char = '●'  # Use a circle character for bubbles
        self.max_output_value = 255

    def get_console_height(self):
        # Get the console height
        return self.console.size.height - 5  # Leave some space for the prompt

    def map_output_to_height(self, output_value):
        # Map output value (0-255) to bubble height (0 - max_height)
        height = int((output_value / self.max_output_value) * self.max_height)
        return height

    def create_bubble_column(self, output_history):
        # Create a column of bubbles for a single output
        column_lines = []
        for output_value in reversed(output_history):
            if output_value == 0:
                line = Text(' ')  # Empty space for silence
            else:
                # Create a bubble with appropriate color and size
                bubble_size = self.map_output_to_height(output_value)
                bubble = Text(self.bubble_char * bubble_size, style="bold cyan")
                line = Align.center(bubble, vertical="middle")
            column_lines.append(line)
        # Fill up the remaining space if history is shorter than max_height
        while len(column_lines) < self.max_height:
            column_lines.append(Text(' '))
        return column_lines

    def display(self, outputs):
        # Update history with the new outputs
        self.history.append(outputs)
        if len(self.history) > self.max_height:
            self.history.pop(0)

        # Transpose history to get per-output histories
        output_histories = list(zip(*self.history))

        # Create columns for each output
        columns = []
        for output_history in output_histories:
            column_lines = self.create_bubble_column(output_history)
            panel = Panel(
                Group(*column_lines),
                border_style="white",
                padding=(0, 1),
                box=None,
            )
            columns.append(panel)

        # Create the layout
        layout = Columns(columns, expand=True)

        # Update the live display
        self.live.update(layout)

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.live.__exit__(exc_type, exc_value, traceback)


class SimpleRichVisualizer(Visualizer):
    def __init__(self, num_outputs, max_height=10):
        super().__init__(num_outputs)
        self.history = []
        self.max_height = max_height
        self.bubble_char = '●'  # Use a circle character for bubbles
        self.max_output_value = 255

    def map_output_to_height(self, output_value):
        # Map output value (0-255) to bubble height (1 - max_height)
        height = int((output_value / self.max_output_value) * (self.max_height - 1)) + 1
        return height

    def create_bubble_column(self, output_history):
        # Create a column of bubbles for a single output
        column_lines = []
        for output_value in reversed(output_history):
            if output_value == 0:
                line = Text(' ')  # Empty space for silence
            else:
                # Create bubble with appropriate size
                bubble_height = self.map_output_to_height(output_value)
                bubble = Text(self.bubble_char * bubble_height, style="bold cyan")
                line = Align.center(bubble)
            column_lines.append(line)
        return column_lines

    def display(self, outputs):
        # Update history with the new outputs
        self.history.append(outputs)
        if len(self.history) > self.max_height:
            self.history.pop(0)

        # Transpose history to get per-output histories
        output_histories = list(zip(*self.history))

        # Create columns for each output
        columns = []
        for output_history in output_histories:
            column_lines = self.create_bubble_column(output_history)
            panel = Panel(
                Group(*column_lines),
                padding=(0, 1),
                box=None,
            )
            columns.append(panel)

        # Create the layout
        layout = Columns(columns)

        # Use Live to update the console
        return layout


class ConsoleVisualizer(Visualizer):
    def __init__(self, num_outputs, max_height=None):
        super().__init__(num_outputs)
        if max_height is None:
            # get from console height
            max_height = Console().size.height - 2
        print("Console height:", max_height)
        self.max_height = max_height

    def display(self, outputs, width=4):
        if width < 4:
            raise ValueError("Width must be at least 4")
        # center the outputs inside the width for each output (given they are width 3)
        front_spacer = ' ' * ((width - 3) // 2)
        back_spacer = ' ' * (width - 3 - len(front_spacer))
        output = ''.join(f'{front_spacer}{output:03d}{back_spacer}' for output in outputs)
        print("_" * len(output))
        print(output)


class BubbleVisualizer(ConsoleVisualizer):
    def __init__(self, num_outputs, max_height=20):
        super().__init__(num_outputs, max_height=max_height)
        self.history = np.zeros((num_outputs, max_height))

    def display(self, outputs):
        # Update history with the new outputs
        self.history = np.roll(self.history, 1, axis=1)
        self.history[:, 0] = outputs

        # Clear the console
        sys.stdout.write("\x1b[2J\x1b[H")
        console_width = Console().size.width
        width_per_output = (console_width - self.num_outputs) // self.num_outputs

        # display history as rows of "bubbles" up to width_per_output
        for row in range(self.max_height - 1, -1, -1):
            line = ' '
            for output in range(self.num_outputs):
                output_value = self.history[output, row]
                bubble_size = int(output_value / 255 * width_per_output)
                bubble = '●' * bubble_size
                # Add spaces to fill the width, centered
                line += ' ' * ((width_per_output - bubble_size) // 2)
                line += bubble
                line += ' ' * ((width_per_output - bubble_size) // 2)
            print(line)

        super().display(outputs, width=width_per_output)


class BarVisualizer(ConsoleVisualizer):
    def __init__(self, num_outputs, max_height=None):
        super().__init__(num_outputs, max_height=max_height)

    def display(self, outputs):
        # Clear the console
        sys.stdout.write("\x1b[2J\x1b[H")
        for row in range(self.max_height - 1, -1, -1):
            line = ' '
            for v in outputs:
                column_height = int(v / 255 * self.max_height)
                if row < column_height:
                    line += '█'  # Filled block
                else:
                    line += ' '  # Empty space
                line += '   '  # Space between columns
            print(line)
        super().display(outputs)


class ListVisualizer(Visualizer):
    def __init__(self, num_outputs, replace=True):
        super().__init__(num_outputs)
        self.replace = replace

    def display(self, outputs):
        output = '   '.join(f'{output:03d}' for output in outputs)
        if max(outputs) ==  255:
            output += '  [MAX]    '
        elif max(outputs) == 0:
            output += '  [SILENCE]'
        else:
            output += '           '
        if self.replace:
            print(output, end='\r') # replace the previous line
        else:
            print(output)


# Test the OutputVisualizer
if __name__ == "__main__":
    console = Console()
    num_outputs = 5
    visualizer = SimpleRichVisualizer(num_outputs)

    test_outputs = [
        [0, 0, 0, 0, 0],
        [50, 100, 150, 200, 255],
        [0, 0, 0, 0, 0],
        [100, 80, 60, 40, 20],
        [255, 200, 150, 100, 50],
        [0, 0, 0, 0, 0],
    ]

    with Live(console=console, refresh_per_second=4) as live:
        for outputs in test_outputs:
            layout = visualizer.display(outputs)
            live.update(layout)
            time.sleep(1)