import tkinter as tk
from time import sleep
from tkinter import ttk
import subprocess
# root window
root = tk.Tk()
root.geometry("240x240")
root.title('15 puzzle solver')
root.resizable(0, 0)

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=1)


#what to do when start is clicked
puzzle = [tk.StringVar() for i in range(16)]
for i, p in enumerate(puzzle):
    p.set((i + 1) % 16)
def start_clicked():
    print([p.get() for p in puzzle])
    f = open('../build/input.txt', 'w')
    f.write('\n'.join(p.get() for p in puzzle))
    f.close()
    sleep(1)
    subprocess.run("./rl")

text = tk.StringVar()
for i in range(16):
    row = int(i/4)
    col = int(i%4)
    username_entry = ttk.Entry(root, textvariable=puzzle[i])
    username_entry.grid(column=col, row=row, padx=5, pady=5)

login_button = ttk.Button(root, text="start", command=start_clicked)
login_button.grid(column = 1, row = 4)
root.mainloop()

