
from Sokoban_env1 import Sokoban_v2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

file_path = 'q_table1.pkl'

with open(file_path, 'rb') as file:
    q_table = pickle.load(file)  

env = Sokoban_v2(map_name="special_3") 

state = env.reset()

frames = []

while True:
    action = np.argmax(q_table[state, :])
    new_state, reward, done, _ = env.step(action)
    img = env.render()
    frames.append(img)
    if done:
        break
    state = new_state
env.close()

# Tạo hoạt hình bằng matplotlib
fig = plt.figure()
im = plt.imshow(frames[0])

def updatefig(i):
    im.set_array(frames[i])
    return [im]

ani = animation.FuncAnimation(fig, updatefig, frames=range(len(frames)), interval=200, blit=True)
plt.show()
