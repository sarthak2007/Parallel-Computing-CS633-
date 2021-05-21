# Execution instructions: python3 plot.py data
# Here "data" file is generated using run.sh 

#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_data(filename):

    data = []
    with open(filename, 'r') as f:
        
        cnt = 0
        for line in f.readlines():

            line = line[:-1]
            if len(line) == 0 or cnt % 5 != 4:
                cnt += 1
                continue
            
            time = float(line.strip())
            data.append(time)

            cnt += 1

    return data

def make_plot(data):
    sns.set()

    demo_input_format = pd.DataFrame.from_dict({
        "P": [],
        "ppn": [],
        "time": [],
    })

    index = 0
    for execution in range(5):
        for P in [1, 2]:
            for ppn in [1, 2, 4]:
                    # Change with the actual data
                    demo_input_format = demo_input_format.append({
                        "P": P, "ppn": ppn, "time": data[index]
                    }, ignore_index=True)

                    index += 1

    type_convert_dict = {
        'P': int,
        'ppn': int,
    }
    demo_input_format = demo_input_format.astype(type_convert_dict)

    demo_input_format["(P, ppn)"] = list(map(lambda x, y: ("(" + x + ", " + y + ")"), map(str, demo_input_format["P"]), map(str, demo_input_format["ppn"])))


    sns.catplot(x="(P, ppn)", y="time", data=demo_input_format, kind="box")

    plt.savefig('plot.jpg')



datafile = sys.argv[1]

data = get_data(datafile)

make_plot(data)
