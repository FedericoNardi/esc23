# include <omp.h>
# include <iostream>

int main()
{
    #pragma omp parallel
    {
        std::cout<<"Hello, ";
        std::cout<<"world!\n";
    }
}