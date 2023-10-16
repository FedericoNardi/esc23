/*

NAME:   
   sum

Purpose: 
   Create an array of numbers and explore floating point 
   arithmetic issues as you sum the array

History: 
   Written by Tim Mattson, 9/2023.

*/

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

#define N 100000  //length of sequence of numbers to work with  

//============================================================================
// support functions
//============================================================================
void seed(float, float);              // seed pseudorandom number generator
float frandom();                      // Get next pseudorandom in the sequence
void FillSequence(double*, double*);    // fill sequence (>0), return "true" sum

//============================================================================

int main ()
{
   double seq[N];    //Sequence to sum, statically defined to keep the code simple
   double True_sum;  //The best estimate of the actual sum

   FillSequence(seq, &True_sum); // populate seq with random values > 0

   std::vector<float> v;
   for(int ii=0; ii<N; ii++){
      v.push_back(seq[ii]);
   }

   // basic, naive sum
   //  Implement your sum here
   //  output result and compare to True_sum
   double sum = 0;
   for(int ii=0; ii<N/2; ii++){
      sum+=seq[ii];
   }


   // std::cout<<"v: { ";
   // for(int ii=0; ii<N; ii++){
   //    std::cout<<v[ii]<<" ";
   // }
   // std::cout<<"}\n";

   std::sort(v.begin(), v.end());

   // std::cout<<"seq: { ";
   // for(int ii=0; ii<N; ii++){
   //    std::cout<<seq[ii]<<" ";
   // }
   // std::cout<<"}\n";

   // std::cout<<"v_sort: { ";
   // for(int ii=0; ii<N; ii++){
   //    std::cout<<v[ii]<<" ";
   // }
   // std::cout<<"}\n";

   double sorted_sum = std::accumulate(v.begin(), v.end(), 0);
   std::cout<<"--> Naive sum: "<<sum<<"\n";
   std::cout<<"--> Sorted sum: "<<sorted_sum<<"\n";
   std::cout<<"--> True sum: "<<True_sum<<"\n";
}


//=================================================================
//  don't worry about (or even look at) anything below this line.  
//=================================================================

//================================================================
//  Function to fill a sequence of N values greater than zero and
//  produce a high quality estimate of the sum of those values
//================================================================
void FillSequence(double *seq, double *True_sum)
{

   float True = 0.0f, val;
   float shift_up   = 100.0f;
   float shift_down =   0.0000000001f;

   seed(0.0, 1.0);
   
   double up_sum   = 0.0d;
   double down_sum = 0.0d;

   for(int i=0;i<N; i++){
     if(i%2==0){
         seq[i]   = (float) frandom() * shift_up;
         up_sum   += (double) seq[i];
     }
     else {
         seq[i]   = (float) frandom() * shift_down;
         down_sum += (double) seq[i];
     }
   }
   
   *True_sum = (float)(up_sum + down_sum);
}
/*

   For maximum portability, this file includes a very simple random number
   generator.  This is not a high quality generator and should not be used
   for serious work.

   The Generator is a linear congruential generator with constants selected
   to yield decent results for sequences with fewer than 2^28 numbers.  The
   pseudo random sequence is seeded with a range

       void seed(lower_limit, higher_limit)
   
   and then subsequent calls to the random number generator generates values
   in the sequence:

       float frandom()
*/

// random number file-scope variables, seed and generator
long random_last = 0.0;
float random_low, random_hi;
void seed(float, float);
float frandom();

//**********************************************************
// Pseudo random number generator:
//     double random
//     void seed (lower_limit, higher_limit)
//**********************************************************
//
// A simple linear congruential random number generator
// (Numerical Recipies chapter 7, 1st ed.) with parameters
// from the table on page 198j.
//
//  Uses a linear congruential generator to return a value between
//  0 and 1, then scales and shifts it to fill the desired range.  This
//  range is set when the random number generator seed is called.
//
static long MULTIPLIER  = 1366;
static long ADDEND      = 150889;
static long PMOD        = 714025;

void seed(float low_in, float hi_in);

float frandom()
{
    long random_next;
    float ret_val;

// 
// compute an integer random number from zero to mod
//
    random_next = (MULTIPLIER  * random_last + ADDEND)% PMOD;
    random_last = random_next;

//
// shift into preset range
//
    ret_val = ((float)random_next/(float)PMOD)*(random_hi-random_low)+random_low;
    return ret_val;
}

//
// set the seed and the range
//
void seed(float low_in, float hi_in)
{
   if(low_in < hi_in)
   { 
      random_low = low_in;
      random_hi  = hi_in;
   }
   else
   {
      random_low = hi_in;
      random_hi  = low_in;
   }
   random_last = PMOD/ADDEND;  // just pick something

}
//**********************************************************
// end of pseudo random generator code.
//**********************************************************

