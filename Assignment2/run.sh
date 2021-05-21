#!/bin/bash

make
rm -f data

for E in `seq 1 10`
do
	for P in 4 16 # num of nodes
	do
        for ppn in 1 8
        do
            groups=2
            python3 script.py $groups $(($P/$groups)) $ppn
            shuf temp_hostfile > hostfile
            rm temp_hostfile

            for D in 16 256 2048 # in KB
            do
                printf "\nE = $E, P = $P, ppn = $ppn, D = $D\n" >> data
                mpirun -np $(($P*$ppn)) -f hostfile ./code $D $ppn $(($P/$groups)) >> data
            done
        done
	done
done

python3 plot.py data 1
