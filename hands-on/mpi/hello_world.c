#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;
    char name[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Getting processor details
    MPI_Get_processor_name(name, &nameLen);
    // Print statements
    printf("Running on %s\n", name);
    printf("--> Process %d of %d | Hello,\n", rank, size);
    printf("--> Process %d of %d | world!\n", rank, size);
    MPI_Finalize();
    return 0;
}