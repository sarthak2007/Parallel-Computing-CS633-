#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

int my_rank, num_procs;
int num_doubles, ppn, nodes_per_grp;
int grp_num = -1, hostname_integer = 0;

// Find and assign the process a group number in the physical topology
// Based on the host it is running on
void find_grp_num_from_node() {
    int length;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Get_processor_name(hostname, &length);

    for (int i = 5; i < length; i++) hostname_integer = 10 * hostname_integer + (hostname[i] - '0');

    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("nodefile.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int line_num = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        // remove '\n' from ending
        line[strlen(line) - 1] = '\0';

        char *pt;
        pt = strtok(line, ",");
        while (pt != NULL) {
            
            // take integer part of the hostname
            int temp_hostname_integer = 0;
            for (int i = 5; i < strlen(pt); i++) temp_hostname_integer = 10 * temp_hostname_integer + (pt[i] - '0');

            if (temp_hostname_integer == hostname_integer) {
                grp_num = line_num;
                break;
            }

            pt = strtok(NULL, ",");
        }

        line_num++;

        if (grp_num != -1) break;
    }
    // printf("Node: %d, Grp: %d\n", hostname_integer, grp_num);

    fclose(fp);
}


typedef struct {
    MPI_Comm intra_node_comm;
    MPI_Comm intra_grp_node_leaders_comm;
    MPI_Comm grp_leaders_comm;

    int my_rank_in_intra_node_comm;
    int my_rank_in_intra_grp_node_leaders_comm;
    int my_rank_in_grp_leaders_comm;
} Comms;

void construct_comms(Comms *local_comms, int root) {
    local_comms -> my_rank_in_intra_node_comm = -1;
    local_comms -> my_rank_in_intra_grp_node_leaders_comm = -1;
    local_comms -> my_rank_in_grp_leaders_comm = -1;

    // Construct an intra-node comm
    // For processing running on a node (ppn)
    // Key for root set to -1 so as to get its rank in new comm as 0!
    MPI_Comm_split(MPI_COMM_WORLD, hostname_integer, (my_rank == root) ? -1 : my_rank, &(local_comms -> intra_node_comm));
    MPI_Comm_rank(local_comms -> intra_node_comm, &(local_comms -> my_rank_in_intra_node_comm));


    // Construct a comm for leaders of each node in a network group
    // rank = 0 in intra-node-comm is made leader in every node
    // Key for root set to -1 so as to get its rank in new comm as 0!
    if (local_comms -> my_rank_in_intra_node_comm == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, grp_num, (my_rank == root) ? -1 : my_rank,
            &(local_comms -> intra_grp_node_leaders_comm));
        MPI_Comm_rank(local_comms -> intra_grp_node_leaders_comm,
            &(local_comms -> my_rank_in_intra_grp_node_leaders_comm));
    } 
    else // Handles the nodes which are excluded in above comm
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, my_rank, &(local_comms -> intra_grp_node_leaders_comm));


    // Construct a comm for leaders of every netowork group
    // Rank = 0 in intra_grp_node_leaders_comm is made leader in this comm
    // Key for root set to -1 so as to get its rank in new comm as 0!
    if (local_comms -> my_rank_in_intra_grp_node_leaders_comm == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, (my_rank == root) ? -1 : my_rank, &(local_comms -> grp_leaders_comm));
        MPI_Comm_rank(local_comms -> grp_leaders_comm, &(local_comms -> my_rank_in_grp_leaders_comm));
    }
    else // Handles the nodes which are excluded in above comm
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, my_rank, &(local_comms -> grp_leaders_comm));

}

void free_comms(Comms *local_comms) {
    MPI_Comm_free(&(local_comms -> intra_node_comm));

    if ((local_comms -> my_rank_in_intra_node_comm) == 0) {

        MPI_Comm_free(&(local_comms -> intra_grp_node_leaders_comm));

        if ((local_comms -> my_rank_in_intra_grp_node_leaders_comm) == 0)
            MPI_Comm_free(&(local_comms -> grp_leaders_comm));
    }

}

void check_correctness(double *data1, double *data2, int size) {
    const double THRESHOLD = 0.0000001;
	int numdiffs = 0;

	for (int i = 0; i < size; i++) {
		double this_diff = data1[i] - data2[i];
        // printf("%d %lf %lf\n", i, data1[i], data2[i]);
		if (fabs(this_diff) > THRESHOLD)
			numdiffs++;
	}

	if (numdiffs > 0)
        printf("%d Diffs found\n", numdiffs);
	// else
		// printf("No diffs found\n");
}

void swap(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}

/********************************************************************
 * MPI_Bcast implementation block starts
 ********************************************************************/

void MPI_Bcast_default(double *data, int root) {
    MPI_Bcast(data, num_doubles, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void MPI_Bcast_optimized(double *data, int root, Comms *local_comms) {
    // Root == Rank0 in group-leaders-comm
    // It broadcasts to other network-group-leaders
    if (local_comms -> my_rank_in_intra_grp_node_leaders_comm == 0)
        MPI_Bcast(data, num_doubles, MPI_DOUBLE, 0, local_comms -> grp_leaders_comm);

    // In Each network-group the leader broadcasts to each node
    if (local_comms -> my_rank_in_intra_node_comm == 0)
        MPI_Bcast(data, num_doubles, MPI_DOUBLE, 0, local_comms -> intra_grp_node_leaders_comm);

    // In each node the leader broadcasts to every process with that node
    MPI_Bcast(data, num_doubles, MPI_DOUBLE, 0, local_comms -> intra_node_comm);

}

void RUN_Bcast() {

	double *data1 = (double*) malloc((num_doubles) * sizeof(double));
	double *data2 = (double*) malloc((num_doubles) * sizeof(double));

    double time_default, time_optimized;
    time_default = time_optimized = 0;

    double time_optimized_comms = 0;

    double start_time, end_time, diff_time, max_time;
    int root = 0;

    Comms local_comms;

    for (int iter = 0; iter < 5; iter++) {

        // Initialise data with random values
        for (int j = 0; j < num_doubles; j++) {
            if (my_rank == root) data1[j] = rand();
            else data1[j] = 0;

            data2[j] = data1[j];
        }

        // Construct sub-communicators and record time
        start_time = MPI_Wtime();
        construct_comms(&local_comms, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
        if (my_rank == root) time_optimized_comms += max_time;

        // Execute default Broadcast (MPI_BCast) and record time
        start_time = MPI_Wtime();
        MPI_Bcast_default(data1, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_default += max_time;

        // Execute Optimised Broadcast and record time
        start_time = MPI_Wtime();
        MPI_Bcast_optimized(data2, root, &local_comms);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_optimized += max_time;

        check_correctness(data1, data2, num_doubles);

        // Free memory for comms
        free_comms(&local_comms);
    }


	if (my_rank == root) {
        printf ("MPI_Bcast_default time = %lf\n", time_default / 5);
	    printf ("MPI_Bcast_optimized time optimized without comms = %lf\n", time_optimized / 5);
	    printf ("MPI_Bcast_optimized time only comms = %lf\n", time_optimized_comms / 5);
	    printf ("MPI_Bcast_optimized time with comms = %lf\n", time_optimized / 5 + time_optimized_comms / 5);
    }

    free(data1);
    free(data2);
}

/********************************************************************
 * MPI_Bcast implementation block ends
 ********************************************************************/

/********************************************************************
 * MPI_Reduce implementation block starts
 ********************************************************************/

void MPI_Reduce_default(double *data_in, double *data_out, int root) {
    MPI_Reduce(data_in, data_out, num_doubles, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
}

void MPI_Reduce_optimized(double **data_in, double **data_out, int root, Comms *local_comms) {

    // swap input and output data to reduce the overhead of allocating and handling temporary memory 
    int cnt_swaps = 0;

    // In each node the leader reduces data from every process within that node
    MPI_Reduce(*data_in, *data_out, num_doubles, MPI_DOUBLE, MPI_MAX, 0, local_comms -> intra_node_comm);

    // output data becomes input data for the next communication
    swap(data_in, data_out);
    cnt_swaps++;

    // In Each network-group the leader reduces from each node
    if (local_comms -> my_rank_in_intra_node_comm == 0) {
        MPI_Reduce(*data_in, *data_out, num_doubles, MPI_DOUBLE, MPI_MAX, 0, local_comms -> intra_grp_node_leaders_comm);
    
        // output data becomes input data for the next communication
        swap(data_in, data_out);
        cnt_swaps++;
    }
        
    // Root == Rank0 in group-leaders-comm
    // It reduces from other network-group-leaders
    if (local_comms -> my_rank_in_intra_grp_node_leaders_comm == 0)
        MPI_Reduce(*data_in, *data_out, num_doubles, MPI_DOUBLE, MPI_MAX, 0, local_comms -> grp_leaders_comm);

    // Retain the data_out pointer to the original state
    if (cnt_swaps == 1)
        swap(data_in, data_out);
}

void RUN_Reduce() {

	double *data_in = (double*) malloc((num_doubles) * sizeof(double));

	double *data1_out = (double*) malloc((num_doubles) * sizeof(double));
	double *data2_out = (double*) malloc((num_doubles) * sizeof(double));

    double time_default, time_optimized;
    time_default = time_optimized = 0;

    double time_optimized_comms = 0;

    double start_time, end_time, diff_time, max_time;
    int root = 0;

    Comms local_comms;

    for (int iter = 0; iter < 5; iter++) {

        // Initialise data with random values
        for (int j = 0; j < num_doubles; j++) {
            data_in[j] = rand();
            data1_out[j] = data2_out[j] = 0;
        }

        // Construct sub-communicators and record time
        start_time = MPI_Wtime();
        construct_comms(&local_comms, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
        if (my_rank == root) time_optimized_comms += max_time;

        // Execute default Reduce (MPI_Reduce) and record time
        start_time = MPI_Wtime();
        MPI_Reduce_default(data_in, data1_out, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_default += max_time;

        // Execute Optimised Reduce and record time
        start_time = MPI_Wtime();
        MPI_Reduce_optimized(&data_in, &data2_out, root, &local_comms);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_optimized += max_time;

        if (my_rank == root) check_correctness(data1_out, data2_out, num_doubles);

        // Free memory for comms
        free_comms(&local_comms);
    }


	if (my_rank == root) {
        printf ("MPI_Reduce_default time = %lf\n", time_default / 5);
	    printf ("MPI_Reduce_optimized time optimized without comms = %lf\n", time_optimized / 5);
	    printf ("MPI_Reduce_optimized time only comms = %lf\n", time_optimized_comms / 5);
	    printf ("MPI_Reduce_optimized time with comms = %lf\n", time_optimized / 5 + time_optimized_comms / 5);
    }

    free(data_in);
    free(data1_out);
    free(data2_out);

}

/********************************************************************
 * MPI_Reduce implementation block ends
 ********************************************************************/

/********************************************************************
 * MPI_Gather implementation block starts
 ********************************************************************/

void MPI_Gather_default(double *data_in, double *data_out, int root) {
    MPI_Gather(data_in, num_doubles, MPI_DOUBLE, data_out, num_doubles, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void MPI_Gather_optimized(double *data_in, double *data_out, int root, Comms *local_comms, double *temp_data_in,
    double *temp_data_out) {

    for (int j = 0; j < num_doubles + 1; j++)
        temp_data_in[j] = data_in[j];

    int data_size = num_doubles + 1;

    // In each node the leader Gathers data from every process within that node
    MPI_Gather(temp_data_in, data_size, MPI_DOUBLE, temp_data_out, data_size, MPI_DOUBLE,
        0, local_comms -> intra_node_comm);

    // output data becomes input data for the next communication
    swap(&temp_data_in, &temp_data_out);
    

    // In Each network-group the leader Gathers from each node
    data_size *= ppn;
    if (local_comms -> my_rank_in_intra_node_comm == 0) {
        MPI_Gather(temp_data_in, data_size, MPI_DOUBLE, temp_data_out, data_size, MPI_DOUBLE,
            0, local_comms -> intra_grp_node_leaders_comm);
    
        // output data becomes input data for the next communication
        swap(&temp_data_in, &temp_data_out);
    }
        
    // Root == Rank0 in group-leaders-comm
    // It Gathers from other network-group-leaders
    data_size *= nodes_per_grp;

    if (local_comms -> my_rank_in_intra_grp_node_leaders_comm == 0) {
            MPI_Gather(temp_data_in, data_size, MPI_DOUBLE, temp_data_out, data_size, MPI_DOUBLE,
            0, local_comms -> grp_leaders_comm);
    }
    
    // Arrange the output in ascending order w.r.t to ranks at the root process
    if (my_rank == root) {
        for (int i = 0; i < (num_doubles + 1) * num_procs; i += num_doubles + 1) {
            int curr_rank = temp_data_out[i + num_doubles];
            for (int j = i; j < i + num_doubles; j++)
                data_out[num_doubles * curr_rank + j - i] = temp_data_out[j];
        }
    }
}

void RUN_Gather() {

    // 1 extra size to incorporate my_rank to order the elements in the root
	double *data_in = (double*) malloc((num_doubles + 1) * sizeof(double));

	double *data1_out = (double*) malloc((num_doubles * num_procs) * sizeof(double));
	double *data2_out = (double*) malloc((num_doubles * num_procs) * sizeof(double));

    double *temp_data_in = (double*) malloc(((num_doubles + 1) * num_procs) * sizeof(double));
	double *temp_data_out = (double*) malloc(((num_doubles + 1) * num_procs) * sizeof(double));

    double time_default, time_optimized;
    time_default = time_optimized = 0;

    double time_optimized_comms = 0;

    double start_time, end_time, diff_time, max_time;
    int root = 0;

    Comms local_comms;

    for (int iter = 0; iter < 5; iter++) {

        // Initialise data with random values
        for (int j = 0; j < num_doubles; j++) {
            data_in[j] = rand();
        }
        // store rank in the last field
        data_in[num_doubles] = my_rank;

        // Construct sub-communicators and record time
        start_time = MPI_Wtime();
        construct_comms(&local_comms, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
        if (my_rank == root) time_optimized_comms += max_time;

        // Execute default Gather (MPI_Gather) and record time
        start_time = MPI_Wtime();
        MPI_Gather_default(data_in, data1_out, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_default += max_time;

        // Execute Optimised Gather and record time
        start_time = MPI_Wtime();
        MPI_Gather_optimized(data_in, data2_out, root, &local_comms, temp_data_in, temp_data_out);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_optimized += max_time;

        if (my_rank == root) check_correctness(data1_out, data2_out, num_doubles * num_procs);

        // Free memory for comms
        free_comms(&local_comms);
    }


	if (my_rank == root) {
        printf ("MPI_Gather_default time = %lf\n", time_default / 5);
	    printf ("MPI_Gather_optimized time optimized without comms = %lf\n", time_optimized / 5);
	    printf ("MPI_Gather_optimized time only comms = %lf\n", time_optimized_comms / 5);
	    printf ("MPI_Gather_optimized time with comms= %lf\n", time_optimized / 5 + time_optimized_comms / 5);
    }

    free(data_in);
    free(data1_out);
    free(data2_out);
    free(temp_data_in);
    free(temp_data_out);

}

/********************************************************************
 * MPI_Gather implementation block ends
 ********************************************************************/

/********************************************************************
 * MPI_Alltoallv implementation block starts
 ********************************************************************/

void MPI_Alltoallv_default(double *data_in, double *data_out, int *send_cnts, int *send_displs,
    int *recv_cnts, int *recv_displs) {

    // construct recv_cnts and recv_displs from send_cnts and send_displs of other processes
    MPI_Alltoall(send_cnts, 1, MPI_INT, recv_cnts, 1, MPI_INT, MPI_COMM_WORLD);

    recv_displs[0] = 0;
    for (int i = 1; i < num_procs; i++) recv_displs[i] = recv_cnts[i-1] + recv_displs[i-1];

    MPI_Alltoallv(&data_in[1], send_cnts, send_displs, MPI_DOUBLE, data_out, recv_cnts, recv_displs,
        MPI_DOUBLE, MPI_COMM_WORLD);
}

void MPI_Gatherv_optimized(double *data_in, double *data_out, Comms *local_comms, int data_size, double *temp_data) {
    // for each network level first the data sizes are gathered by root from each process
    // then the data is gathered by the root

    for (int j = 0; j < data_size; j++)
        temp_data[j] = data_in[j];

    int *recv_cnts = (int*) malloc(num_procs * sizeof(int));
    int *recv_displs = (int*) malloc(num_procs * sizeof(int));

    // In each node the leader gathers data from every process within that node
    MPI_Gather(&data_size, 1, MPI_INT, recv_cnts, 1, MPI_INT, 0, local_comms -> intra_node_comm);

    if (local_comms -> my_rank_in_intra_node_comm == 0) {
        recv_displs[0] = 0;
        for (int i = 1; i < ppn; i++) recv_displs[i] = recv_displs[i-1] + recv_cnts[i-1];
    }

    MPI_Gatherv(temp_data, data_size, MPI_DOUBLE, data_out, recv_cnts, recv_displs, MPI_DOUBLE,
        0, local_comms -> intra_node_comm);

    // output data becomes input data for the next communication
    swap(&temp_data, &data_out);
    

    // In Each network-group the leader gathers from each node
    if (local_comms -> my_rank_in_intra_node_comm == 0) {
        data_size = recv_displs[ppn - 1] + recv_cnts[ppn - 1];

        MPI_Gather(&data_size, 1, MPI_INT, recv_cnts, 1, MPI_INT, 0, local_comms -> intra_grp_node_leaders_comm);
    
        if (local_comms -> my_rank_in_intra_grp_node_leaders_comm == 0) {
            recv_displs[0] = 0;
            for (int i = 1; i < nodes_per_grp; i++) recv_displs[i] = recv_displs[i-1] + recv_cnts[i-1];
        }

        MPI_Gatherv(temp_data, data_size, MPI_DOUBLE, data_out, recv_cnts, recv_displs, MPI_DOUBLE,
            0, local_comms -> intra_grp_node_leaders_comm);
    
        // output data becomes input data for the next communication
        swap(&temp_data, &data_out);
    }
        
    // Root == Rank0 in group-leaders-comm
    // It gathers from other network-group-leaders

    if (local_comms -> my_rank_in_intra_grp_node_leaders_comm == 0) {
        data_size = recv_displs[nodes_per_grp - 1] + recv_cnts[nodes_per_grp - 1];

        MPI_Gather(&data_size, 1, MPI_INT, recv_cnts, 1, MPI_INT, 0, local_comms -> grp_leaders_comm);
    
        if (local_comms -> my_rank_in_grp_leaders_comm == 0) {
            recv_displs[0] = 0;
            int groups = num_procs / (ppn * nodes_per_grp);
            for (int i = 1; i < groups; i++) recv_displs[i] = recv_displs[i-1] + recv_cnts[i-1];
        }

        MPI_Gatherv(temp_data, data_size, MPI_DOUBLE, data_out, recv_cnts, recv_displs, MPI_DOUBLE,
        0, local_comms -> grp_leaders_comm);
    }
}

void MPI_Alltoallv_optimized(double *data_in, double *data_out, Comms *local_comms,
    int *send_cnts, int *send_displs, double *temp_data_in, double *temp_data_out, double *temp_data_for_gatherv) {

    int root = 0;
    int *all_send_cnts = (int*) malloc((num_procs * num_procs) * sizeof(int));
    int *all_send_displs = (int*) malloc((num_procs * num_procs) * sizeof(int));
    
    // gather all the send counts and displacements at rank 0
    MPI_Gather(send_cnts, num_procs, MPI_INT, all_send_cnts, num_procs, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(send_displs, num_procs, MPI_INT, all_send_displs, num_procs, MPI_INT, root, MPI_COMM_WORLD);

    // gather all the data at rank 0
    int size_to_send = send_cnts[num_procs - 1] + send_displs[num_procs - 1] + 1;
    MPI_Gatherv_optimized(data_in, temp_data_in, local_comms, size_to_send, temp_data_for_gatherv);
    
    int *final_send_cnts = (int*) malloc(num_procs * sizeof(int));
    int *final_send_displs = (int*) malloc(num_procs * sizeof(int));

    // convert the data at root into suitable format for scattering
    if (my_rank == root) {
        int curr = 0;
        for (int i = 0; i < num_procs; i++) {
            int rank = temp_data_in[curr];
            int end = curr + all_send_displs[(rank * num_procs) + num_procs - 1] +
                all_send_cnts[(rank * num_procs) + num_procs - 1] + 1;
            for (int j = curr + 1; j < end; j++) {
                temp_data_out[rank * num_doubles + (j - curr - 1)] = temp_data_in[j];
            }
            curr = end;
        }

        int index = 0;
        for (int dest = 0; dest < num_procs; dest++) {
            final_send_displs[dest] = index;

            for (int src = 0; src < num_procs; src++) {
                int start = (src * num_doubles) + all_send_displs[src * num_procs + dest];
                int cnt = all_send_cnts[src * num_procs + dest];
                for (int i = start; i < start + cnt; i++)
                    temp_data_in[index++] = temp_data_out[i];
            }

            final_send_cnts[dest] = index - final_send_displs[dest];
        }
    }

    // send number of elements to be received to each process
    int recv_cnt;
    MPI_Scatter(final_send_cnts, 1, MPI_INT, &recv_cnt, 1, MPI_INT, root, MPI_COMM_WORLD);

    // send the corresponding data to each process
    MPI_Scatterv(temp_data_in, final_send_cnts, final_send_displs, MPI_DOUBLE, data_out, recv_cnt,
        MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void RUN_Alltoallv() {
    int data_size = (rand() % num_doubles) + 1;

    if (data_size % num_procs != 0)
        data_size += (num_procs - data_size % num_procs);

    // 1 extra size to incorporate my_rank to maintain the order of the elements
	double *data_in = (double*) malloc((data_size + 1) * sizeof(double));

	double *data1_out = (double*) malloc((num_doubles * num_procs) * sizeof(double));
	double *data2_out = (double*) malloc((num_doubles * num_procs) * sizeof(double));

    // Set send counts and send displacements
    int *send_cnts = (int*) malloc(num_procs * sizeof(int));
    int *send_displs = (int*) malloc(num_procs * sizeof(int));
    for (int i = 0; i < num_procs; i++) {
        send_cnts[i] = data_size / num_procs;
        send_displs[i] = i * send_cnts[i];
    }
    int *recv_cnts = (int*) malloc(num_procs * sizeof(int));
    int *recv_displs = (int*) malloc(num_procs * sizeof(int));

    double *temp_data_in = (double*) malloc(((num_doubles + 1) * num_procs) * sizeof(double));
	double *temp_data_out = (double*) malloc(((num_doubles + 1) * num_procs) * sizeof(double));
	double *temp_data_for_gatherv = (double*) malloc(((num_doubles + 1) * num_procs) * sizeof(double));

    double time_default, time_optimized;
    time_default = time_optimized = 0;

    double time_optimized_comms = 0;

    double start_time, end_time, diff_time, max_time;
    int root = 0;

    Comms local_comms;

    for (int iter = 0; iter < 5; iter++) {

        // store rank in the first field
        data_in[0] = my_rank;

        // Initialise data with random values
        for (int j = 1; j < data_size + 1; j++) {
            data_in[j] = rand();
        }

        // Construct sub-communicators and record time
        start_time = MPI_Wtime();
        construct_comms(&local_comms, root);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
        if (my_rank == root) time_optimized_comms += max_time;

        // Execute default Alltoallv (MPI_Alltoallv) and record time
        start_time = MPI_Wtime();
        MPI_Alltoallv_default(data_in, data1_out, send_cnts, send_displs, recv_cnts, recv_displs);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_default += max_time;

        // Execute Optimised Alltoallv and record time
        start_time = MPI_Wtime();
        MPI_Alltoallv_optimized(data_in, data2_out, &local_comms, send_cnts, send_displs, temp_data_in,
            temp_data_out, temp_data_for_gatherv);
        end_time = MPI_Wtime();

        diff_time = end_time - start_time;
        MPI_Reduce (&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        if (my_rank == root) time_optimized += max_time;

        check_correctness(data1_out, data2_out, recv_cnts[num_procs - 1] + recv_displs[num_procs - 1]);

        // Free memory for comms
        free_comms(&local_comms);
    }


	if (my_rank == root) {
        printf ("MPI_Alltoallv_default time = %lf\n", time_default / 5);
	    printf ("MPI_Alltoallv_optimized time without comms = %lf\n", time_optimized / 5);
	    printf ("MPI_Alltoallv_optimized time only comms = %lf\n", time_optimized_comms / 5);
	    printf ("MPI_Alltoallv_optimized time with comms = %lf\n", time_optimized / 5 + time_optimized_comms / 5);
    }

    free(data_in);
    free(data1_out);
    free(data2_out);

    free(send_cnts);
    free(send_displs);
    free(recv_cnts);
    free(recv_displs);

    free(temp_data_in);
    free(temp_data_out);
    free(temp_data_for_gatherv);

}

/********************************************************************
 * MPI_Alltoallv implementation block ends
 ********************************************************************/

int main(int argc, char *argv[]) {

	int size_of_data_KB = atoi(argv[1]);
    num_doubles = size_of_data_KB * (1024 / 8);

    ppn = atoi(argv[2]);
    nodes_per_grp = atoi(argv[3]);

    MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	srand((my_rank + 1) * time(0));

    // populate node to group number mapping from nodefile.txt
    find_grp_num_from_node();

    RUN_Bcast();
    RUN_Reduce();
    RUN_Gather();
    RUN_Alltoallv();

	MPI_Finalize();

    return 0;
}
