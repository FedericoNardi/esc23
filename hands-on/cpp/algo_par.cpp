#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <chrono>
#include <cassert>

int main()
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> uniform_dist{1, MAX_N};

  // fill a vector with SIZE random numbers
  int const SIZE = 1'000'000'000;
  std::vector<int> v;
  v.reserve(SIZE);
  std::generate_n(std::back_inserter(v), SIZE, [&] { return uniform_dist(eng); });

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::accumulate
    auto sum = std::accumulate(v.begin(),v.end(),0ll); // 0ll: 0 of type long long
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout<<"--> Sum: "<<sum<<"\n";
    std::chrono::duration<float> d = t1 - t0;
    std::cout<<"Accumulate";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, sequential policy
    auto sum = std::reduce(v.begin(),v.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout<<"--> Sum: "<<sum<<"\n";
    std::chrono::duration<float> d = t1 - t0;
    std::cout<<"Reduce";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, parallel policy
    auto sum = std::reduce(std::execution::par, v.begin(), v.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout<<"--> Sum: "<<sum<<"\n";
    std::chrono::duration<float> d = t1 - t0;
    std::cout<<"Reduce with parallel policy";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto copy = v;
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort
    std::sort(copy.begin(),copy.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout<<"Sort";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto copy = v;
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort, sequential policy
    std::sort(std::execution::seq,copy.begin(),copy.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout<<"Sort - sequential";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto copy = v;
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort, parallel policy
    std::sort(std::execution::par,copy.begin(),copy.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout<<"Sort - parallel";
    std::cout << " in " << d.count() << " s\n";
  }
}
