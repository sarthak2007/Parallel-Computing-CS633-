#!/bin/bash

make
rm -f data

for E in `seq 1 5`
do
	for P in 16 36 49 64
	do
		~/UGP/allocator/src/allocator.out $P 8 > log
		for N in 256 1024 4096 16384 65536 262144 1048576
		do
			echo "E = $E, P = $P, N = $N" >> data
			mpirun -np $P -f hosts ./halo $N 50 >> data
		done
	done
done
