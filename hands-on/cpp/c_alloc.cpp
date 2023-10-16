#include <cstdlib>
#include <iostream>
#include <memory>

void do_something_with(int* p, int size);

int main()
{
  // allocate memory for 1000 int's
  int const SIZE = 1000;
  // auto p = static_cast<int*>(std::malloc(SIZE * sizeof(int)));

  
  std::shared_ptr<void> p = {
    std::malloc(SIZE * sizeof(int)),
    [](auto ptr){std::free(ptr);}
  };
  

  //std::cout<<*p<<"\n";
  do_something_with(static_cast<int*>(p.get()), SIZE);
}

void do_something_with(int* p, int size)
{
  std::fill(p, p + size, 42);
}