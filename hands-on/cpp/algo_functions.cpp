#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // multiply all the elements of the vector
  // use std::accumulate
  auto product = std::accumulate(v.begin(), v.end(), 1LL,
    [](auto v1, auto v2){return v1*v2;}
  );
  std::cout<<"---> Product: "<<product<<"\n";

  // compute the mean and the standard deviation
  // use std::accumulate and a struct with two numbers to accumulate both the sum and the sum of squares

  // lambda takes a pair and an integer, and the returns in the pair the accumulated values
  // PROVA A FARLO CON make_pair
  struct mystruct {
    int sum;
    int sum_squares;
  };
  auto out = std::accumulate(
    v.begin(), v.end(),mystruct{0,0},
    [](mystruct blocks, int v2){ 
      blocks.sum+=v2;
      blocks.sum_squares+=v2*v2;
      return blocks;
    }
  );
  
  std::cout<<"---> Mean: "<<1.*out.sum/N<<"\n";
  std::cout<<"---> Std : "<<std::sqrt(1.*out.sum_squares/(N)-pow(1.*out.sum/N,2))<<"\n";
  {
    auto copy = v;
    // sort the vector in descending order
    // use std::sort
    std::sort(copy.begin(), copy.end(),
        [](auto x1, auto x2){return x1>x2;}
    );
    std::cout<<"---> Sorted: "<<copy<<"\n";
  }

  // move the even numbers at the beginning of the vector
  // use std::partition
  {
    auto copy = v;
    std::partition(copy.begin(), copy.end(), [](auto i){return i%2==0;});
    std::cout<<"---> Partitioned: "<<copy<<"\n";
  }

  // create another vector with the squares of the numbers in the first vector
  // use std::transform
  {
    auto copy = v;
    std::transform(copy.begin(),copy.end(),copy.begin(),[](auto x){return x*x;});
    std::cout<<"---> Original: "<<v<<"\n";
    std::cout<<"---> Squared : "<<copy<<"\n";
  }

  // find the first multiple of 3 or 7
  // use std::find_if
  {
    auto copy = v;
    auto found = std::find_if(copy.begin(), copy.end(), [](auto x){return (x%7==0 || x%3==0);});
    std::cout<<"---> Multiples of 3 or 7: "<<copy<<"\n";
    std::cout<<"---> First multiple of 3 or 7 at position "<<std::distance(std::begin(v),found)<<"\n";
  }

  // erase from the vector all the multiples of 3 or 7
  // use std::remove_if followed by vector::erase
  //   or the newer std::erase_if utility (C++20)
}

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c)
{
  os << "{ ";
  std::copy(
            std::begin(c),
            std::end(c),
            std::ostream_iterator<int>{os, " "}
            );
  os << '}';

  return os;
}

std::vector<int> make_vector(int N)
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&] { return dist(eng); });

  return result;
}
