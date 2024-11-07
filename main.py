import os
import sys
import multiprocessing as mp
import time


if __name__ == "__main__":

    # 切换到文件所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    list_M = [0.4]
    list_P_pump = [50e-3]
    list_FSR = [25e9]

    pool_simulation = []
    pool_plot = []
    list_dir_name = []
    nums = len(list_M) * len(list_P_pump) * len(list_FSR)
    for FSR in list_FSR:
        for P_pump in list_P_pump:
            for M in list_M:
                dir_name = time.strftime("%Y%m%d:%H%M:%S", time.localtime())
                list_dir_name.append(dir_name)
                command = f"python simulation.py {M} {P_pump} {FSR} {dir_name}"
                pool_simulation.append(
                    mp.Process(target=os.system, args=(command,))
                    )
                # pool_plot.append(
                #     mp.Process(
                #         target=os.system, args=(
                #             f"python plot.py {M} {P_pump} {FSR} {dir_name}",
                #             )
                #         )
                #     )
                time.sleep(1.2)
                print(f"{len(pool_simulation)}/{nums} constructed, " + command)

    length = len(pool_simulation)
    parallel = 4
    for i in range(0, length, parallel):
        for j in range(parallel):
            if i+j >= length:
                break
            try:
                pool_simulation[i+j].start()
            except Exception as e:
                print(e)
        for j in range(parallel):
            if i+j >= length:
                break
            try:
                pool_simulation[i+j].join()
                # pool_plot[i+j].start()
                # pool_plot[i+j].join()
            except Exception as e:
                print(e)
            print(f"Simulation {i+j+1}/{length} finished")
