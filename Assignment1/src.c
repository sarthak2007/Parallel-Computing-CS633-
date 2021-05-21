#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void swap(double **prev, double **curr) {
	double *temp = *prev;
	*prev = *curr;
	*curr = temp;
}

void initialise(double *data, double *data_upd, int n) {
	for(int i = 0; i < n+2; i++) {
		for(int j = 0; j < n+2; j++) {
			if(i == 0 || j == 0 || i == n+1 || j == n+1) data[i*(n+2) + j] = 0;
			data_upd[i*(n+2) + j] = 0;
		}
	}
}

// Send and Recv for different methods
void send_recv(double *data, int send_start, int recv_start, int stride, int count, 
				int time_stamp, int method, int comm_proc_rank, MPI_Request *request, 
				int *no_of_requests, double *recv_buffer, int n) {

	// Send each element separately
	if (method == 0) {
		for(int i = send_start, cnt = 0; i < (send_start + count * stride); i += stride, cnt++) {
			MPI_Isend(&data[i], 1, MPI_DOUBLE, comm_proc_rank, cnt, MPI_COMM_WORLD, &request[(*no_of_requests)++]);
		}
		for(int i = recv_start, cnt = 0; i < (recv_start + count * stride); i += stride, cnt++) {
			MPI_Irecv(&data[i], 1, MPI_DOUBLE, comm_proc_rank, cnt, MPI_COMM_WORLD, &request[(*no_of_requests)++]);
		}
	} 
	// PACK and send data
	else if(method == 1) {
		double send_buffer[n];
		int position = 0; 

		for(int i = send_start; i < (send_start + count * stride); i += stride) {
			MPI_Pack(&data[i], 1, MPI_DOUBLE, send_buffer, 8*n, &position, MPI_COMM_WORLD);
		}

		MPI_Isend(send_buffer, position, MPI_PACKED, comm_proc_rank, 0, MPI_COMM_WORLD, &request[(*no_of_requests)++]);
		MPI_Irecv(recv_buffer, 8*n, MPI_PACKED, comm_proc_rank, 0, MPI_COMM_WORLD, &request[(*no_of_requests)++]);
	} 
	// Send data using vector type
	else if(method == 2) {
		MPI_Datatype newvtype;

		MPI_Type_vector(count, 1, stride, MPI_DOUBLE, &newvtype);
		MPI_Type_commit(&newvtype);

		MPI_Isend(&data[send_start], 1, newvtype, comm_proc_rank, 0, MPI_COMM_WORLD, &request[(*no_of_requests)++]);
		MPI_Irecv(&data[recv_start], 1, newvtype, comm_proc_rank, 0, MPI_COMM_WORLD, &request[(*no_of_requests)++]);
	}
}

double compute(double *data, int i, int j, int p, int p_row, int p_col, int n) {
	double val = 0;
	int neighbors = 0;
	// Top
	if((i == 1 && p_row != 0) || i > 1) val += data[(i-1)*(n+2) + j], neighbors++;
	// Left
	if((j == 1 && p_col != 0) || j > 1) val += data[i*(n+2) + j - 1], neighbors++;
	// Down
	if((i == n && p_row != p-1) || i < n) val += data[(i+1)*(n+2) + j], neighbors++;
	// Right
	if((j == n && p_col != p-1) || j < n) val += data[i*(n+2) + j + 1], neighbors++;

	// Take Avg
	val /= neighbors;
	return val;
}


void send_compute_recv_async(double *data, double *data_upd, int n, int p, int my_rank, int method, MPI_Request *request, int time_stamp) {
	// Process's row/col in processes grid
	int p_row = my_rank / p;
	int p_col = my_rank % p;

	// Keep track of number of requests sent + recv
	int no_of_requests = 0;

	// Communicate with neighboring processes
	// top process
	if (p_row != 0) {
		MPI_Isend(&data[1*(n+2) + 1], n, MPI_DOUBLE, my_rank-p, time_stamp, MPI_COMM_WORLD, &request[no_of_requests++]);
		MPI_Irecv(&data[0*(n+2) + 1], n, MPI_DOUBLE, my_rank-p, time_stamp, MPI_COMM_WORLD, &request[no_of_requests++]);
	}
	// down process
	if (p_row != p-1) {
		MPI_Isend(&data[(n)*(n+2) + 1], n, MPI_DOUBLE, my_rank+p, time_stamp, MPI_COMM_WORLD, &request[no_of_requests++]);
		MPI_Irecv(&data[(n+1)*(n+2) + 1], n, MPI_DOUBLE, my_rank+p, time_stamp, MPI_COMM_WORLD, &request[no_of_requests++]);
	}

	// Receive buffer to be used to store data to be UNPACKED in method2
	double recv_buffer[2][n];

	// left process
	if (p_col != 0) send_recv(data, 1*(n+2) + 1, 1*(n+2) + 0, n+2, n, time_stamp, method, my_rank-1, request, &no_of_requests, recv_buffer[0], n);
	// right process
	if (p_col != p-1) send_recv(data, 1*(n+2) + n, 1*(n+2) + n+1, n+2, n, time_stamp, method, my_rank+1, request, &no_of_requests, recv_buffer[1], n);

	// Computation of inner grid (non-halo region)
	for(int i = 2; i < n; ++i) {
		for(int j = 2; j < n; ++j) {
			data_upd[i*(n+2) + j] = (data[(i-1)*(n+2) + j] + data[(i+1)*(n+2) + j] +
									 data[i*(n+2) + j + 1] + data[i*(n+2) + j - 1]) / 4;
		}
	}

	// Wait for all requests
	MPI_Waitall(no_of_requests, request, MPI_STATUSES_IGNORE);

	// In case of second method, UNPACK the data
	if(method == 1) {
		int position = 0;
		for(int i = (n+2); i < (n+1)*(n+2); i += n+2) {
			MPI_Unpack(recv_buffer[0], 8*n, &position, &data[i], 1, MPI_DOUBLE, MPI_COMM_WORLD);
		}
		position = 0;
		for(int i = (n+2) + n+1; i < (((n+2) + n+1) + (n+2)*n); i += n+2) {
			MPI_Unpack(recv_buffer[1], 8*n, &position, &data[i], 1, MPI_DOUBLE, MPI_COMM_WORLD);
		}
	}
}

void halo_exchange(double *data, double *data_upd, int n, int p, int time_stamp, int my_rank, int method) {
	// Process's row/col in processes grid
	int p_row = my_rank / p;
	int p_col = my_rank % p;

	MPI_Request request[8 * n];

	while(time_stamp--) {
		initialise(data, data_upd, n);

		// Commute between processes and perform inner-grid computation as well
		send_compute_recv_async(data, data_upd, n, p, my_rank, method, request, time_stamp);

		// TESTING
		// for(int i = 0; i < p*p; i++) {
		//     MPI_Barrier(MPI_COMM_WORLD);
		//     if (i == my_rank) {
		//         printf("Printing data for rank = %d at time_stamp = %d \n", my_rank, time_stamp);
		// 		for(int i = 0; i < n+2; ++i) {
		// 			for(int j = 0; j < n+2; ++j) {
		// 				printf("%lf ", data[i*(n+2) + j]);
		// 			}
		// 			printf("\n");
		// 		}
		// 		printf("\n\n");
		//     }
		// }

		// Compute for the halo regions
		for(int i = 1; i < n+1; ++i) {
			for(int j = 1; j < n+1; ++j) {
				// Check if it is a halo point
				if(i == 1 || i == n || j == 1 || j == n) {
					data_upd[i*(n+2) + j] = compute(data, i, j, p, p_row, p_col, n);
				}
			}
		}

		// Swap data so that now data contains the updated data.
		swap(&data, &data_upd);
	}
}

int main(int argc, char *argv[]) {
	
	int my_rank, size;
	double start_time[3], end_time[3], diff_time[3], max_time[3];

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Take size of data and final_time_stamp as i/p
	int size_of_data = atoi(argv[1]);
	int final_time_stamp = atoi(argv[2]);

	// Dim of square data matrix
	int n = sqrt(size_of_data);
	// Dim of square processes matrix
	int p = sqrt(size);

	// Will be used to store data from (1,1) to (n,n)
	// 2 extra rows and cols will be used to store 
	// data got neighboring processes
	double *data = (double*) malloc((n+2)*(n+2)*sizeof(double));
	// This will be used to store computed values for each
	// time stamp, i.e. the new matrix at every time stamp
	double *data_upd = (double*) malloc((n+2)*(n+2)*sizeof(double));

	// Initialising data with random values
	srand(my_rank * time(0));
	for(int i = 0; i < n+2; ++i) {
		for(int j = 0; j < n+2; ++j) {
			 if(i == 0 || i == n+1) data[i*(n+2) + j] = 0;
			 else if(j == 0 || j == n+1) data[i*(n+2) + j] = 0;
			 else data[i*(n+2) + j] = rand(); //%10; 
		}
	}

	// Perform Halo Exchange using Method 1
	start_time[0] = MPI_Wtime();
   	halo_exchange(data, data_upd, n, p, final_time_stamp, my_rank, 0);
  	end_time[0] = MPI_Wtime();
  	diff_time[0] = end_time[0] - start_time[0];

	// Obtain max time over all processes
	MPI_Reduce (&diff_time[0], &max_time[0], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!my_rank) printf ("%lf\n", max_time[0]);

	// Perform Halo Exchange using Method 2
	start_time[1] = MPI_Wtime();
   	halo_exchange(data, data_upd, n, p, final_time_stamp, my_rank, 1);
  	end_time[1] = MPI_Wtime();
  	diff_time[1] = end_time[1] - start_time[1];

	// Obtain max time over all processes
	MPI_Reduce (&diff_time[1], &max_time[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!my_rank) printf ("%lf\n", max_time[1]);

	// Perform Halo Exchange using Method 3
	start_time[2] = MPI_Wtime();
   	halo_exchange(data, data_upd, n, p, final_time_stamp, my_rank, 2);
  	end_time[2] = MPI_Wtime();
  	diff_time[2] = end_time[2] - start_time[2];

	// Obtain max time over all processes
	MPI_Reduce (&diff_time[2], &max_time[2], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!my_rank) printf ("%lf\n", max_time[2]);

	MPI_Finalize();
	return 0;
}