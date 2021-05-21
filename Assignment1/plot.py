import matplotlib.pyplot as plt
import sys
import math

def get_data(filename):

    data = {}
    with open(filename, 'r') as f:
        
        cnt = 0
        for line in f.readlines():

            line = line[:-1]
            if cnt%4 == 0:
                cnt %= 4
                P, N = [int(x.split('=')[-1].strip()) for x in line.split(',')[1:]]
                N = int(math.sqrt(N))
            else:
                if P not in data:
                    data[P] = {}
                if N not in data[P]:
                    data[P][N] = [[], [], []]
                data[P][N][cnt-1].append(float(line))
            cnt += 1
        
    return data

def make_plot(data, P):
    data = list(data.items())
    data.sort()

    N = [x[0] for x in data]
    Y = [x[1] for x in data]
    Y = list(map(list, zip(*Y)))

    Y_mean = list(map(lambda y: [sum(x)/len(x) for x in y], Y))

    colors = ['blue', 'red', 'green']
    method_names = ['Normal', 'Packed', 'Derived']
    widths = [2**(math.log2(p)+.1)-2**(math.log2(p)-.1) for p in N]

    plt.clf()
    for method in range(3):

        c = colors[method]
        plt.plot(N, Y_mean[method], label=method_names[method], color=c)

        plt.boxplot(Y[method], positions=N, widths=widths,
                    boxprops=dict(color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c)
                    )
    

    plt.xlabel('Data Points(sqrt(N))')
    # Set the y axis label of the current axis.
    plt.ylabel('Time taken(in secs)')
    # Set a title of the current axes.

    plt.title('Box plot displaying the performance of the given 3 methods for P = {}'.format(P))
    # show a legend on the plot
    plt.legend()

    plt.yscale('log')
    plt.xscale('log', basex=2)
    # Display a figure.
    # plt.show()

    plt.savefig('plot{}.jpg'.format(P))



data = get_data(sys.argv[1])

for P, per_proc_data in data.items():
    make_plot(per_proc_data, P)