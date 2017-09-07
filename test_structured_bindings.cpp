/*#include <iostream>
#include <utility>			// std::tuple
#include <cmath>

std::pair<float, int> frexp(float arg) {
  int exponent;
  float mantissa = std::frexp(arg, &exponent);
  return {mantissa, exponent};
}


int main() {
	auto [mantissa, exponent] = frexp(0.23f);
	std::cout << mantissa << std::endl;
}
*/
#include <set>
#include <string>
#include <iomanip>
#include <iostream>
 
int main() {
    std::set<std::string> myset;
    if (auto [iter, success] = myset.insert("Hello"); success) 
        std::cout << "insert is successful. The value is " << std::quoted(*iter) << '\n';
    else
        std::cout << "The value " << std::quoted(*iter) << " already exists in the set\n";
}