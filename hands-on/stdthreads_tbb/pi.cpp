#include <iostream>
#include <iomanip>
#include <chrono>
#include <oneapi/tbb.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <atomic>

int main(){
    int num_steps = 1<<22;
    double pi = 0.;
    double step = 1./(double) num_steps;
    double sum = 0.;
    std::atomic_ref<double> sum_atomic(sum);

    unsigned int n_threads = oneapi::tbb::info::default_concurrency();

    auto start = std::chrono::steady_clock::now();

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range(0,num_steps),
        [&](auto& my_range){
            auto local_sum = 0.;
            for(auto i=my_range.begin(); i!=my_range.end(); i++){
                auto x = (i+0.5)*step;
                local_sum += 4./(1.+x*x);
            }
            sum_atomic += local_sum;
        }
    );

    pi = step*sum;

    std::chrono::duration<double> dur = std::chrono::steady_clock::now() - start;
    std::cout << "Time spent in reduction: " << dur.count() << " seconds\n";
    std::cout << "Number of threads: " << n_threads << "\n";

    std::cout<<"--> Result: "<<std::setprecision(15) << pi <<std::endl;
    
}