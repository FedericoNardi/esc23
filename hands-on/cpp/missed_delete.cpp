# include <memory>
# include <iostream>

using SomeType = int;

SomeType* factory();
std::unique_ptr<SomeType> factory_smart();

void do_something(SomeType*);
// void do_something(std::unique_ptr<SomeType>);

int main()
{
  auto t = factory_smart();

  // try {
  auto ptr = t.release();
  do_something(ptr);

  delete ptr;

  // } catch (...) {
  // }
}

SomeType* factory()
{
  return new SomeType{};
}

std::unique_ptr<SomeType> factory_smart(){
  return std::make_unique<SomeType>();
}

void do_something(SomeType* t)
//void do_something(std::unique_ptr<SomeType>)
{
  // throw 1;
  std::cout<<*t<<"\n";
}
