#include <iostream>
#include <numeric>

int main () {
   int numbers[5];

   std::iota (numbers,numbers+5,10);

   std::cout << "numbers are :";
   for (int& i:numbers) std::cout << ' ' << i;
   std::cout << '\n';

   return 0;
}