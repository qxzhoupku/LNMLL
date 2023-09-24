import os
import sys
import multiprocessing as mp

list_M = [0.1, 0.2, 0.3, 0.4]
list_P_pump = [10, 20, 100]
pool = []

for M in list_M:
    for P_pump in list_P_pump:
        pool.append(mp.Process(target=os.system, args=(f"python main.py {M} {P_pump}",)))

for p in pool:
    p.start()

for p in pool:
    p.join()
