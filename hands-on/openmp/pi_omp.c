/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

The is the original sequential program.  It uses the timer
from the OpenMP runtime library

History: Written by Tim Mattson, 11/99.

*/
#include <stdio.h>
#include <omp.h>
# define N_THREADS 8

static long num_steps = 100000000;
double step;
int main ()
{	
	{
	  double sum[N_THREADS] {0.};
	  double start_time, run_time;

	  step = 1.0/(double) num_steps;

	  omp_set_num_threads(N_THREADS);
        	 
	  start_time = omp_get_wtime();
	  # pragma omp parallel
	  {
		int thread_ID = omp_get_thread_num();
		int n_threads = omp_get_num_threads();
		double x = 0;
	    for (int i=0;i< num_steps; i=i+n_threads){
	  	  x = (i+0.5)*step;
	  	  sum[thread_ID] = sum[thread_ID] + 4.0/(1.0+x*x);
	    }
	  }
	  double pi = 0;
	  for(int ii=0; ii<N_THREADS; ii++){
		pi += step*sum[ii];
	  } 
	  //printf("PI = %f", pi);
	  run_time = omp_get_wtime() - start_time;
	  printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time);
	}
	{
	  // double sum[N_THREADS] {0.};
	  double sum {0}, pi {0};
	  double start_time, run_time;

	  step = 1.0/(double) num_steps;

	  omp_set_num_threads(N_THREADS);
        	 
	  start_time = omp_get_wtime();
	  # pragma omp parallel
	  {
		// int thread_ID = omp_get_thread_num();
		int n_threads = omp_get_num_threads();
		double x = 0;
		double local_sum = 0.;
	    for (int i=0;i< num_steps; i=i+n_threads){
	  	  x = (i+0.5)*step;
	  	  // sum[thread_ID] = sum[thread_ID] + 4.0/(1.0+x*x);
		  local_sum += 4./(1.+x*x);
	    }
		// DO PARTIAL SUM INTO A LOCAL VARIABLE AND THEN DO THE TOTAL SUM, OTHERWISE I SERIALIZE THE LOOP
		# pragma omp critical 
			sum += local_sum; 
	  }
	  pi = sum*step;
	  // for(int ii=0; ii<N_THREADS; ii++){
	  // 	pi += step*sum[ii];
	  // } 
	  //printf("PI = %f", pi);
	  run_time = omp_get_wtime() - start_time;
	  printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time);
	}
}	  
