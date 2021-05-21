#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_data(filename, comms_or_not):

    data = {
        'Bcast': [],
        'Reduce': [],
        'Gather': [],
        'Alltoallv': []
    }
    with open(filename, 'r') as f:
        
        cnt = 0
        for line in f.readlines():

            line = line[:-1]
            if len(line) == 0 or line[0] != 'M':
                continue
            
            method = line.split('_')[1]
            time = float(line.split('=')[-1].strip())

            if cnt%4 == 0:
                time_def = time
            elif cnt%4 == 2 * comms_or_not + 1:
                data[method].append((time_def, time))

            cnt += 1

    # for x, y in data.items():
    #     print(x)
    #     for z in y:
    #         print(z[0], z[1])
    return data

def make_plot(data, X):
    sns.set()

    demo_input_format = pd.DataFrame.from_dict({
        "D": [],
        "P": [],
        "ppn": [],
        "mode": [],  # 1 --> optimized, 0 --> standard
        "time": [],
    })

    index = 0
    for execution in range(10):
        for P in [4, 16]:
            for ppn in [1, 8]:
                for D in [16, 256, 2048]:
                    # Change with the actual data
                    demo_input_format = demo_input_format.append({
                        "D": D, "P": P, "ppn": ppn, "mode": 1, "time": data[index][1]
                    }, ignore_index=True)
                    demo_input_format = demo_input_format.append({
                        "D": D, "P": P, "ppn": ppn, "mode": 0, "time": data[index][0]
                    }, ignore_index=True)

                    index += 1

    type_convert_dict = {
        'D': int,
        'P': int,
        'ppn': int,
        'mode': int
    }
    demo_input_format = demo_input_format.astype(type_convert_dict)

    demo_input_format["(P, ppn)"] = list(map(lambda x, y: ("(" + x + ", " + y + ")"), map(str, demo_input_format["P"]), map(str, demo_input_format["ppn"])))

    # print(demo_input_format)

    sns.catplot(x="(P, ppn)", y="time", data=demo_input_format, kind="box", col="D", hue="mode")

    plt.savefig('plot_{}.jpg'.format(X))



datafile = sys.argv[1]

# 1 if time of constructing comms should be taken for the optimized versions of the methods else 0
comms_or_not = int(sys.argv[2])
data = get_data(datafile, comms_or_not)

for X, per_proc_data in data.items():
    make_plot(per_proc_data, X)
