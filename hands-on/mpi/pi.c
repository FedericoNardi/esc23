/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

This is the sequenctial version of the program.  It uses
the OpenMP timer.

History: Written by Tim Mattson, 11/99.

*/
#include <stdio.h>
#include <omp.h>
#include <mpi.h>

static long num_steps = 100000000;
double step;
int main(int argc, char **argv)
{
   int i, rank, size;
   double x, pi, buffer, sum = 0.0;
   double start_time, run_time;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   step = 1.0 / (double)num_steps;

   start_time = MPI_Wtime();

   for (i = 0; i <= num_steps; i += size)
   {
      x = (i + rank + 0.5) * step;
      buffer = buffer + 4.0 / (1.0 + x * x);
   }
   // printf("Sbuffer %lf\n", buffer * step);

   MPI_Reduce(&buffer, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   if (rank == 0)
   {
      pi = step * sum;
      run_time = MPI_Wtime() - start_time;
      printf("\n pi with %ld steps is %lf in %lf seconds\n ",
             num_steps, pi, run_time);
   }

   MPI_Finalize();
}
