#include <memory>
#include <cstdlib>
#include <cstring>

char* some_api();

int main()
{
  auto* p = some_api();

  // std::free(p);
  // delete p;
  // delete [] p;
}


#include <cstring>

char* some_api() {
  static char s[] = "Hello, world!";
  return std::strstr(s, "orl");
<<<<<<< HEAD
}
=======
}
>>>>>>> 25bab66d796f94a93ae42057eeaafb2d14be179c
