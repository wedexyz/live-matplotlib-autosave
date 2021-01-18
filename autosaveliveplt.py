import matplotlib.pyplot as plt
import os
import time
import threading

plt.ion()
x=[]
y=[]
script_dir =os.path.dirname(__file__)
result_dir = os.path.join(script_dir,f"{int(time.time())}")
sample_file_name = "tes.jpg"
for i in range(0, 100):
    x.append(i)
    y.append(i+2)
    plt.plot(x,y, 'g-',linewidth=1.5, markersize=4)
    plt.show()
    plt.pause(0.1)
plt.savefig(result_dir+sample_file_name)
