#include <iostream>
#include <chrono>
#include <cassert>
#include <utility>
#include <cstdlib>

using Duration = std::chrono::duration<float>; // former type def: Duration is the type equal to std::chrono::duration<float>
                                               // typedef(chrono::duration<float>, Duration)
                                               // 'using' can be used with templates

std::pair<double, Duration> pi(int n)
{
  assert(n > 0);

  auto const start = std::chrono::high_resolution_clock::now();

  auto const step = 1. / n;
  auto sum = 0.;
  for (int i = 0; i != n; ++i) {
    auto x = (i + 0.5) * step;
    sum += 4. / (1. + x * x);
  }

  auto const end = std::chrono::high_resolution_clock::now();

  return { step * sum, end - start };
}

int main(int argc, char* argv[])
{
  int const n = (argc > 1) ? std::atoi(argv[1]) : 10;

  auto const [value, time] = pi(n); //Structure-binding -> unpacks

  std::cout << "pi = " << value
            << " for " << n << " iterations"
            << " in " << time.count() << " s\n";
}
