import os
import sys
import multiprocessing as mp
import time

# 切换到文件所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

list_M = [0.5, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
list_P_pump = [20e-3, 200e-3]
list_FSR = [25e9, 10e9]

pool_simulation = []
pool_plot = []
list_dir_name = []
for M in list_M:
    for P_pump in list_P_pump:
        for FSR in list_FSR:
            dir_name = time.strftime("%Y%m%d:%H%M:%S", time.localtime())
            list_dir_name.append(dir_name)
            pool_simulation.append(mp.Process(target=os.system, args=(f"python simulation.py {M} {P_pump} {FSR} {dir_name}",)))
            # pool_plot.append(mp.Process(target=os.system, args=(f"python plot.py {M} {P_pump} {FSR} {dir_name}",)))
            time.sleep(2)

if __name__ == "__main__":
    length = len(pool_simulation)
    for i in range(length):
        try:
            pool_simulation[i].start()
            pool_simulation[i].join()
            # pool_plot[i].start()
            # pool_plot[i].join()
        except:
            pass
        print(f"Simulation {i+1}/{length} finished")