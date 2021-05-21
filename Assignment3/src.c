#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

int my_rank, num_procs;
float *temperature_data = NULL;
int num_years = -2, num_stations = 0;

// Populate temperature data from tdata.csv
void fetch_data(char *filename) {

    FILE *fp = fopen(filename, "r");

    char *line = NULL;
    size_t len = 0;

    int line_num = 0, curr_num_rows_in_data = 0;
    int curr_data_index = 0;

    while (getline(&line, &len, fp) != -1) {
        // remove '\n' from ending
        line[strlen(line) - 1] = '\0';

        // allocate more memory if the data size is small
        if (line_num > curr_num_rows_in_data) {
            if (curr_num_rows_in_data == 0) curr_num_rows_in_data = 1;
            else curr_num_rows_in_data *= 2;

            float *allocated_data = realloc(temperature_data, (curr_num_rows_in_data * num_years) * sizeof(float));
            if (allocated_data != NULL) temperature_data = allocated_data;
        }

        char *pt;
        int cnt = 0;

        pt = strtok(line, ",");
        while (pt != NULL) {
            cnt++;    

            if (line_num == 0) num_years++;

            // skip first 2 cols
            else if(line_num != 0 && cnt > 2)
                // convert string to float
                sscanf(pt, "%f", &temperature_data[curr_data_index++]);

            pt = strtok(NULL, ",");
        }
        line_num++;
    }
    num_stations = line_num - 1;

    // for (int i = 0; i < num_stations; i++) {
    //     for (int j = 0; j < num_years; j++)
    //         printf("%f ", temperature_data[i * num_years + j]);
    //     printf("\n");
    // }

    fclose(fp);
}


// partition data w.r.t rows i.e. allocate some rows with all the columns to different processes
float* strategy1() {
    // send num of years and stations to all the processes
    MPI_Bcast(&num_years, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_stations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // scatter rows to different processes
    float *recv_buf = (float*) malloc((ceil(num_stations * 1.0 / num_procs) * num_years) * sizeof(float));
    MPI_Scatter(temperature_data, num_stations/num_procs * num_years, MPI_FLOAT, recv_buf,
        num_stations/num_procs * num_years, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // find minimum for each process
    float *min_temp_per_year_per_proc = (float*) malloc((num_years) * sizeof(float));
    for (int i = 0; i < num_years; i++) min_temp_per_year_per_proc[i] = 1e6;

    for (int i = 0; i < num_stations/num_procs; i++) {
        for (int j = 0; j < num_years; j++) {
            min_temp_per_year_per_proc[j] = fminf(min_temp_per_year_per_proc[j], recv_buf[i * num_years + j]);
        }
    }

    // root will find minimum temperatures for the remaining rows that were not scattered
    if (my_rank == 0) {
        for (int i = num_procs * (num_stations / num_procs); i < num_stations; i++) {
            for (int j = 0; j < num_years; j++) {
                min_temp_per_year_per_proc[j] = fminf(min_temp_per_year_per_proc[j], 
                    temperature_data[i * num_years + j]);
            }
        }
    }

    // min temperatures per year over all the processes
    float *min_temp_per_year = (float*) malloc((num_years) * sizeof(float));
    for (int i = 0; i < num_years; i++) min_temp_per_year[i] = 1e6;

    MPI_Reduce(min_temp_per_year_per_proc, min_temp_per_year, num_years, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

    return min_temp_per_year;

}

// partition data w.r.t columns i.e. allocate some columns with all the rows to different processes
float* strategy2() {
    // send num of years and stations to all the processes
    MPI_Bcast(&num_years, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_stations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Set block size, only last process block size can be smaller/larger
    int block_size = (my_rank == (num_procs-1)) ? (num_years - (num_years / num_procs)*my_rank) : (num_years / num_procs);

    // New vector type
    MPI_Datatype newvtype;
    MPI_Type_vector(num_stations, num_years / num_procs, num_years, MPI_FLOAT, &newvtype);
    MPI_Type_commit(&newvtype);

    // New vector type for last process
    MPI_Datatype newvtype2;
    MPI_Type_vector(num_stations, (num_years - (num_years / num_procs)*(num_procs-1)), num_years, MPI_FLOAT, &newvtype2);
    MPI_Type_commit(&newvtype2);

    MPI_Request request[num_procs];
    int num_of_requests = 0;

    float *recv_data = (float*) malloc((num_stations * block_size) * sizeof(float));

    // Distribute columns to other processes
    if(my_rank == 0) {
        for(int i = 0; i < num_procs; ++i) {
            MPI_Isend(&temperature_data[i*(num_years/num_procs)], 1, (i == num_procs-1) ? newvtype2 : newvtype, i, i, MPI_COMM_WORLD, &request[(num_of_requests++)]);
        }
    }

    MPI_Irecv(recv_data, num_stations * block_size, MPI_FLOAT, 0, my_rank, MPI_COMM_WORLD, &request[(num_of_requests++)]);

    // Wait for all requests
    MPI_Waitall(num_of_requests, request, MPI_STATUSES_IGNORE);

    // find minimum for each process
    float *min_temp_per_year_per_proc = (float*) malloc((block_size) * sizeof(float));
    for (int i = 0; i < block_size; i++) min_temp_per_year_per_proc[i] = 1e6;

    for (int i = 0; i < num_stations; i++) {
        for (int j = 0; j < block_size; j++) {
            min_temp_per_year_per_proc[j] = fminf(min_temp_per_year_per_proc[j], recv_data[i * block_size + j]);
        }
    }

    // Assemble the min of years from every process into the root process
    int recvcounts[num_procs], displs[num_procs];
    for(int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i == (num_procs-1)) ? (num_years - (num_years / num_procs)*i) : (num_years / num_procs);
        if(i == 0) displs[i] = 0;
        else displs[i] = displs[i-1] + recvcounts[i-1];
    }

    // min temperatures per year over all the processes
    float *min_temp_per_year = (float*) malloc((num_years) * sizeof(float));
    
    MPI_Gatherv(min_temp_per_year_per_proc, block_size, MPI_FLOAT, min_temp_per_year, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    return min_temp_per_year;

}

float* find_minimum_temp_per_year_serially() {
    float *min_temp_per_year = (float*) malloc((num_years) * sizeof(float));
    for (int i = 0; i < num_years; i++) min_temp_per_year[i] = 1e6;

    for (int i = 0; i < num_stations; i++) {
        for (int j = 0; j < num_years; j++) {
            min_temp_per_year[j] = fminf(min_temp_per_year[j], temperature_data[i * num_years + j]);
        }
    }

    return min_temp_per_year;
}

void check_correctness(float *data1, float *data2, int size) {
    const float THRESHOLD = 0.0000001;
	int numdiffs = 0;

	for (int i = 0; i < size; i++) {
		float this_diff = data1[i] - data2[i];
		if (fabs(this_diff) > THRESHOLD) {
            // printf("%d %f %f\n", i, data1[i], data2[i]);
			numdiffs++;
        }
	}

	if (numdiffs > 0)
        printf("%d Diffs found\n", numdiffs);
	// else
	// 	printf("No diffs found\n");
}


int main(int argc, char *argv[]) {

    char *filename = argv[1];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // populate the temperature data from tdata.csv
    if (my_rank == 0) fetch_data(filename);

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time, end_time, diff_time, max_time;

    start_time = MPI_Wtime();

    // Distribute data to all processes and compute minimum temperatures

    // call strategy2() if you want to run it
    float *min_temp_per_year = strategy1();
    float overall_min_temp = 1e6;

    if (my_rank == 0) {
        for (int i = 0; i < num_years; i++) overall_min_temp = fminf(overall_min_temp, min_temp_per_year[i]);
    }

    end_time = MPI_Wtime();

    diff_time = end_time - start_time;
    MPI_Reduce(&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        // find all the minimas serially by root and check correctness
        float *min_temp_per_year_serial = find_minimum_temp_per_year_serially();
        check_correctness(min_temp_per_year, min_temp_per_year_serial, num_years);
        
        for (int i = 0; i < num_years - 1; i++) printf("%0.2f,", min_temp_per_year[i]);
        printf("%0.2f\n", min_temp_per_year[num_years - 1]);
        printf("%0.2f\n", overall_min_temp);

        printf("%lf\n", max_time);

    }

    MPI_Finalize();

    return 0;
}
