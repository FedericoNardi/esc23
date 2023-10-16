// === ABOUT TEMPLATES ===

template<int M>
class LessThan {
    public:
        auto operator()(int n) const {return n<M; }
};

LessThan<42> lt42{};
auto b1 = lt42(32);
LessThan<24> lt24{};
auto b2 = lt24(32);

// Now the int is part of the type -> Saves a few memory bytes

//===========================

// NOTE: A POINTER IS AN ITERATOR
#include <algorithm>
#include <iostream>

auto f(int* p, int N){
    std::sort(p, p+N);
}

int main(){
    int N = 5;
    int a[] = {4, 10, 54, 23, 42};
    f(a, N);
    std::cout<<"{";
    for(int i=0; i<N; i++){
        std::cout<<a[i]<<" ";
    }
    std::cout<<"}\n";
}

// ==========================

// ===== for range =====
# include <iostream>

auto f(std::vector<int> const& v){
    for( int i=0; i != v.size(); ++i ){
        v[i];
    }
    for (auto& e : v){  //note: reference is important, otherwise I have a copy.
                        //  Also, if I now modify e, I do not modify the vector element 
        e; // e is an alias for the element in the range
    }
}
// ==========================

// ===== Tag dispatching + overload =====
# include <vector>
# include <list>
# include <iostream>

template <class It>
auto __distance(It first, It last, std::random_access_iterator_tag tag){
    return last - first;
}

template <class It>
typename std::iterator_traits<It>::difference_type
__distance(It first, It last, std::random_access_iterator_tag tag){
    typename std::iterator_traits<It>::difference_type n=0;
    while (first != last) {++first; ++n;}
    return n;
}

template <class It>
auto distancee(It first, It last){
    return __distance(first, last, typename std::iterator_traits<It>::iterator_category{});
}

int main(){
    std::vector<int> v(10);
    std::list<int> l(20);

    //std::cout << distancee(v.begin(), v.end());
    std::cout << distancee(l.begin(), l.end());
}
// =========================================

// ========= Computation through template at compile time ===========
// FACTORIAL WITH TEMPLATES
# include <array>

template <int N>
struct F
{
    static const int value = N * F<N-1>::value; // Recursive - functional programming
    // This will never stop decreasing N until the compiler is tired of instantiating the next template
};
// Special case -> Define terminal state
// Can specialize template for F<0>
template <>
struct F<0>
{
    static const int value = 1;
};

static_assert(F<5>::value==120);
std::array<char, F<5>::value> buffer;

// =================================