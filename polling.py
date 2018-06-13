import numpy as np
import tkinter as tk
from tkinter import messagebox

# Model configuration {
N_LINES = 5
MAX_Q, MAX_I = 10, 10
UTILIZATION = 0.6  # arrival rate / service rate
C_HOLDING = 1
C_BACKLOGGING = 2
K_II = 1
K_IJ = 20
K_0I = 100
BOUNDARY_PENALTY = 5000
LONGTERM_BONUS = 0
MAX_STEPS = 100
# }

# Don't change the configuration below {
UNIT = 30
UPPER_BOUND = MAX_Q
LOWER_BOUND = MAX_I
BOUND = UPPER_BOUND + LOWER_BOUND
# }


class Polling(tk.Tk):
    def __init__(self):
        np.random.seed(1)
        super(Polling, self).__init__()
        self.action_space = np.arange(N_LINES + 1).reshape(N_LINES + 1, 1)  # 0 means null action
        self.state_space = np.array(list(zip(*(x.flat for x in np.meshgrid(*(range(-MAX_Q, MAX_I + 1) for i in range(N_LINES))))))).reshape(-1, N_LINES)
        self.geometry('{}x{}'.format(N_LINES * UNIT, BOUND * UNIT + 55))
        self.score = 0
        self.play_mode = False
        self.title('polling')
        self.last_action = 0

        [self.columnconfigure(i, weight=1) for i in range(N_LINES)]
        [tk.Label(self, text=str(i + 1), fg='white', bg='black').grid(row=0, column=i, sticky='WE') for i in range(N_LINES)]
        self.canvas = tk.Canvas(self, bg='white', height=BOUND * UNIT + 5, width=N_LINES * UNIT)
        [self.canvas.create_line(i * UNIT, 0, i * UNIT, BOUND * UNIT + 5) for i in range(0, N_LINES)]
        self.canvas.create_line(0, UPPER_BOUND * UNIT + 5, N_LINES * UNIT, UPPER_BOUND * UNIT + 5, fill='grey', dash=(2, 4))
        self.init_lines()
        self.canvas.grid(row=1, columnspan=N_LINES)

        self.var = tk.StringVar()
        self.var.set('Score: {:.1f}'.format(self.score))
        self.score_board = tk.Label(self, textvariable=self.var, fg='white', bg='black')
        self.score_board.grid(row=2, columnspan=N_LINES, sticky='WE')
        self.steps = 0
        self.max_steps = MAX_STEPS

        if self.play_mode:
            self.bind('<Key>', self.step)

    def init_lines(self):
        self.lines = [None] * N_LINES
        for i in range(N_LINES):
            self.lines[i] = self.canvas.create_line(i * UNIT, UPPER_BOUND * UNIT + 5,
                                                    (i + 1) * UNIT, UPPER_BOUND * UNIT + 5,
                                                    fill='red', width=5)

    def reset(self):
        self.update()
        [self.canvas.delete(l) for l in self.lines]
        self.init_lines()
        self.score = 0
        self.steps = 0
        observation = np.zeros(N_LINES)
        return observation

    def end_game(self, y):
        if self.play_mode and (min(y) <= 0 or max(y) >= BOUND * UNIT):
            messagebox.showinfo('polling', 'Game Over')
            self.destroy()
            return True
        return False

    def c(self, y):
        # non-fixed cost function with inventory vector y (positive=holding, negative=backlogging)
        return C_BACKLOGGING * np.maximum(-y, 0).sum() + C_HOLDING * np.maximum(y, 0).sum()

    def k(self, action, last_action):
        # fixed cost function
        if action == 0:
            return 0
        elif action == last_action:
            return K_II
        elif last_action:
            return K_IJ
        else:
            return K_0I

    def step(self, event):
        # check if the game is ended
        y = np.array([self.canvas.coords(l)[1] - 5 for l in self.lines])
        if self.end_game(y): return

        # arrival
        for i in range(N_LINES):
            line = self.lines[i]
            dy = min(int(np.random.poisson(UTILIZATION / N_LINES)) * UNIT, self.canvas.coords(line)[1] - 5)  # poisson a, deterministic s
            self.canvas.move(line, 0, -dy)

        # service
        action = int(event.keysym) if self.play_mode else event
        if action: self.canvas.move(self.lines[action - 1], 0, UNIT)
        y = np.array([self.canvas.coords(l)[1] - 5 for l in self.lines])
        y_shifted = y // UNIT - UPPER_BOUND
        reward = -self.c(y_shifted) - self.k(action, self.last_action)
        end = any(np.isin(y_shifted, [-UPPER_BOUND, LOWER_BOUND]))
        if end:
            reward -= BOUNDARY_PENALTY
        if self.steps > 0 and self.steps % 10 == 0:
            reward += LONGTERM_BONUS

        # return observation
        observation = np.minimum(np.maximum(y_shifted, -MAX_Q), MAX_I)
        self.score += reward
        end |= ((self.steps == self.max_steps) and not self.play_mode)
        self.var.set('Score: {:.1f}'.format(self.score))
        self.last_action = action
        self.steps += 1
        return observation, reward, end

    def render(self):
        self.update()

    def run(self, f):
        self.after(1, f)
        self.mainloop()


if __name__ == '__main__':
    env = Polling()
    env.play_mode = True
    env.mainloop()
