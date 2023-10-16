# include <memory>
# include <iostream>

std::unique_ptr<int> factory(); //unique_ptr template

// "still reachable"
auto g = factory();

int main()
{
  // "definitely lost"
  auto t = factory();
  std::cout<<*t<<"\n";
}

std::unique_ptr<int> factory()
{
  // return new int;
  return std::make_unique<int>(42); //make_unique is a FUNCTION
}
