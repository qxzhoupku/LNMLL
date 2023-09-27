import os
import sys
import multiprocessing as mp

list_M = [0.1, 0.2, 0.4, 0.5, 0.8, 1.0]
# list_M = [0.1]
list_P_pump = [10e-3, 20e-3, 100e-3, 200e-3]
# list_P_pump = [10]

pool = []
for M in list_M:
    for P_pump in list_P_pump:
        pool.append(mp.Process(target=os.system, args=(f"python simulation.py {M} {P_pump}",)))

if __name__ == "__main__":
    for p in pool:
        p.start()

    for p in pool:
        p.join()
