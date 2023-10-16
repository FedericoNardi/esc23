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

  std::random_device rd;   // a seed source for the random number engine
  std::mt19937 engine(1);  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> uniformDist(-5, 5);
  for (unsigned int i = 0; i < numElements; ++i)
    input.emplace_back(uniformDist(engine));
  // create mutex
  std::mutex mtx;
  long long int sum = 0;
  std::atomic_ref<long long int> sumAtomic(sum);

  auto start = std::chrono::steady_clock::now();

  oneapi::tbb::parallel_for(
    oneeapi::tbb::blocked_range(input.begin(), input.end()), 
    [&](auto& range){
      sumAtomic = 0;
      for (auto& element : range){
        local_sum+=element;
      }
    sumAtomic+=local_sum;
    }
  );

  std::chrono::duration<double> dur = std::chrono::steady_clock::now() - start;
  std::cout << "Time spent in reduction: " << dur.count() << " seconds"
            << std::endl;
  std::cout << "Sum result: " << sum << std::endl;
  return 0;
}
