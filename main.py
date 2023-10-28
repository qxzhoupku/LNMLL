import os
import sys
import multiprocessing as mp
import time

list_M = [0.2, 0.4, 0.6, 0.8]
list_P_pump = [20e-3]
list_FSR = [25e9]

pool = []
for M in list_M:
    for P_pump in list_P_pump:
        for FSR in list_FSR:
            pool.append(mp.Process(target=os.system, args=(f"python simulation.py {M} {P_pump} {FSR}",)))

if __name__ == "__main__":
    for p in pool:
        p.start()
        time.sleep(2)

    for p in pool:
        p.join()
