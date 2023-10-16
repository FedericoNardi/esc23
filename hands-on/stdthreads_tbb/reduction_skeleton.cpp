#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>

int main() {
  const unsigned int numElements = 1E8;

  std::vector<int> input;
  input.reserve(numElements);
  unsigned int n_threads = 2;

  std::vector<std::thread> v;

  std::random_device rd;   // a seed source for the random number engine
  std::mt19937 engine(1);  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> uniformDist(-5, 5);
  for (unsigned int i = 0; i < numElements; ++i)
    input.emplace_back(uniformDist(engine));
  // create mutex
  std::mutex mtx;
  long long int sum = 0;
  std::atomic_ref<long long int> sumAtomic(sum);

  auto f = [&](unsigned long long firstIndex, unsigned long long lastIndex){
    // long long int local_sum = 0;
    // for(auto it = firstIndex; it < lastIndex; it++){
    //   local_sum += input[it];
    // }
    long long int local_sum = std::accumulate(input.begin()+firstIndex, input.begin()+lastIndex, 0.); 
    sumAtomic+=local_sum;
    //std::scoped_lock lock(mtx);
    //sum += local_sum;
  };

  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<n_threads; i++){
    auto myFirstIndex = i*numElements/n_threads;
    auto myLastIndex = (i+1)*numElements/n_threads;
    v.emplace_back(f, myFirstIndex, myLastIndex);
  }

  for(auto& t:v){
    t.join();
  }

  std::chrono::duration<double> dur = std::chrono::steady_clock::now() - start;
  std::cout << "Time spent in reduction: " << dur.count() << " seconds"
            << std::endl;
  std::cout << "Sum result: " << sum << std::endl;
  return 0;
}
