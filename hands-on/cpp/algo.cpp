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

  // sum all the elements of the vector
  // use std::accumulate
  auto sum = std::accumulate(v.begin(),v.end(),0);
  std::cout<<"--> Sum of v: "<<sum<<"\n";

  // compute the average of the first half and of the second half of the vector
  auto size = v.size();
  auto mid_it = v.begin()+size/2;
  std::cout<<"--> First avg: "<<std::reduce(v.begin(),mid_it)/(1.*std::distance(v.begin(),mid_it))<<"\n";
  std::cout<<"--> Second avg: "<<std::reduce(mid_it,v.end())/(1.*std::distance(mid_it,v.end()))<<"\n";

  // move the three central elements to the beginning of the vector
  // use std::rotate
  std::rotate(v.begin(),v.begin()+N/2-1,v.begin()+N/2+2); //rotate(first,middle,last)
  std::cout<<"--> Rotated vector: "<<v<<"\n";

  // remove duplicate elements
  // use std::sort followed by std::unique/unique_copy
  std::vector<int> d = {4, 12, 73, 42, 12, 1, 73, 4, 73, 12, 9, 12, 73, 12};
  std::cout<<"New test vector: "<<d<<"\n";
  std::sort(d.begin(), d.end());
  std::cout<<"--> Sorted vector: "<<d<<"\n";
  std::vector<int> d1;
  std::unique_copy(d.begin(),d.end(),std::back_inserter(d1));
  std::cout<<"--> Unique vector: "<<d1<<"\n";
  // unique overwrite and returns iterator to last significant element. Then can erase the rest

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
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&] { return dist(eng); });

  return result;
}
