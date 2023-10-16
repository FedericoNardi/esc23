#include <iostream>
#include <iomanip>
#include <chrono>
#include <oneapi/tbb.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <random>

//
// The monte carlo pi program
//

static long num_trials = 1E8;

int main()
{
   long int count_in = 0;
   double pi, x, y, test;
   double r = 1.0; // radius of circle. Side of squrare is 2*r

   std::random_device rd;  // a seed source for the random number engine
   std::mt19937 engine(1); // mersenne_twister_engine seeded with rd()
   std::uniform_real_distribution<double> uniformDist(-1, 1);

   auto start = std::chrono::steady_clock::now();
   /*
   {
      std::atomic_ref<long int> count_atomic(count_in);

      oneapi::tbb::parallel_for(
          oneapi::tbb::blocked_range(0, static_cast<int>(num_trials)),
          [&](auto &my_range)
          {
             auto tmp_count = 0;
             for (auto i = my_range.begin(); i != my_range.end(); ++i)
             {
                x = uniformDist(engine);
                y = uniformDist(engine);
                test = x * x + y * y;
                if (test <= r * r)
                   tmp_count++;
             }
             count_atomic += tmp_count;
          });
   }
   */
   {
       oneapi::tbb::parallel_for(
         oneapi::tbb::blocked_range(0, static_cast<int>(num_trials)), 
      )
   }

   pi = 4.0 * (static_cast<double>(count_in) / static_cast<double>(num_trials));
   std::chrono::duration<double> dur = std::chrono::steady_clock::now() - start;
   printf("\n %ld trials, pi is %lf ", num_trials, pi);

   std::cout << "--> Time elapsed: " << dur.count() << "\n";
}
