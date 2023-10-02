/*

This program explores solutions to the quadratic equation
using floating point arithmetic.  Specifically, find the 
two values, x, that solve the equation

      a*x^2 + b*x + c = 0

usage:  

      quad a b c

If run without command line arguments, we consider a case 
with serious floating point arithmetic issues

History: Written by Tim Mattson, 09/2023.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
   float a, b, c, radTerm;
   float x1,x2,x1r,x1i,x2r,x2i;
   float t1, t2, t1r, t1i, t2r, t2i;

   if(argc==4) {
     a = (float) atof(*++argv);
     b = (float) atof(*++argv);
     c = (float) atof(*++argv);
   }
   else{
     a = 0.0005f;  
     b = 100.0f;
     c = 0.005f;
   }
   printf(" quad will run with with a = %f, b = %f, and c = %f\n",a,b,c);

   float discriminant = b*b-4.0f*a*c;
   float denom = 2.0f * a;

   // solutions are complex numbers
   if(discriminant < 0.0f){
       radTerm = (float) sqrt(-discriminant);
       x1r = -b/denom;
       x1i = radTerm/denom;
       x2r = -b/denom;
       x2i = -radTerm/denom;
       printf("solutions are %f+%fi and %f+%fi\n",x1r,x1i,x2r,x2i);
       t1r= a*(x1r*x1r-x1i*x1i) + b*x1r  + c;
       t1i= a*(2.0f*x1r*x1i)    + b*x1i;
       t2r= a*(x2r*x2r-x2i*x2i) + b*x2r  + c;
       t2i= a*(2.0f*x2r*x2i)    + b*x2i;
       printf("test results (should be zero) are %f+%fi and %f+%fi\n",t1r,t1i,t2r,t2i);
   }
   // solutions are  a real number
   else{
       radTerm = (float) sqrt(discriminant);
       x1 = (-b+radTerm)/denom;
       x2 = (-b-radTerm)/denom;
       printf("solutions are %f %f\n",x1,x2);
       t1 = a*x1*x1 + b*x1  +  c;
       t2 = a*x2*x2 + b*x2  +  c;
       printf("test results (should be zero) are %f %f\n",t1,t2);
   }

}
