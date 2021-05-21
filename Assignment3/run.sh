#!/bin/bash

make
rm -f output.txt data

for E in `seq 1 5`
do
	for P in 1 2 # num of nodes
	do
        for ppn in 1 2 4
        do
            groups=1
            python3 script.py $groups $(($P/$groups)) $ppn

            printf "\nE = $E, P = $P, ppn = $ppn\n" >> data
            mpirun -np $(($P*$ppn)) -f hostfile ./code tdata.csv > output.txt

            cat output.txt >> data
        done
	done
done

python3 plot.py data
