#include <mpi.h>
#include <stdio.h>

#define VAL 42
#define NREPS 2
#define TAG 5

int main(int argc, char **argv)
{
    int rank, size;
    int n_pass = 10;
    int launch_ball = VAL;
    int recv_ball = 0;
    double start;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status stat;
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        start = MPI_Wtime();
    }

    for (int i = 0; i < n_pass; i++)
    {
        if (rank == 0)
        {
            MPI_Send(&launch_ball, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD);
            MPI_Recv(&recv_ball, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &stat);
            recv_ball = 0;
            printf("PING\n");
        }
        if (rank == 1)
        {
            MPI_Recv(&recv_ball, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &stat);
            MPI_Send(&launch_ball, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
            printf("PONG\n");
        }
    }
    if (rank == 0)
    {
        double end = MPI_Wtime() - start;
        printf("Time elapsed: %f\n", end);
    }

    MPI_Finalize();

    return 0;
}