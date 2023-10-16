#include <thread>
#include <iostream>
#include <vector>
#include <mutex>

int main()
{   
    auto f = [](int i){
        std::cout<<"Hello world from thread "<<i<<"\n";
    };
    {
        std::thread t0(f,0); // ask thread 0 to call function f with argument 0
        t0.join(); // "Wait for t0 to complete and then destroy iy"
    }
    {
        // Need more threads?
        auto n = std::thread::hardware_concurrency();
        std::vector<std::thread> v;
        
        for (auto i=0; i<n; i++){
            v.emplace_back(f,i); // emplace inizializza direttamente all'interno del vector
        }
        for (auto& t:v){
            t.join();
        }
    }
}