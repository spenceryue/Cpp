#ifndef PRINT_VECTOR_H
#define PRINT_VECTOR_H
#include <iostream>				// std::ostream
#include <string>				// std::string
#include <vector>				// std::vector


/* Source: http://en.cppreference.com/w/cpp/container/vector/vector */
template<typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
	s.put('[');
	char comma[3] = {'\0', ' ', '\0'};
	for (const auto& e : v) {
		s << comma << e;
		comma[0] = ',';
	}
	return s << ']';
}

#endif /* PRINT_VECTOR_H */



/* Test print_vector */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream>				// std::cout, std::endl
	#include <string>				// std::string
	#include "faces.h"
	#include "basename.h"
	#include "type_stuff.h"

using namespace std;
using namespace string_literals;
int main(int argc, char* argv[])
{
	cout << pikachu << endl;
	cout << endl << endl;
	
	startup_msg(argv[0]);
	cout << endl << endl;

	// c++11 initializer list syntax:
	std::vector<std::string> words1 {"the", "frogurt", "is", "also", "cursed"};
	std::cout << "words1: " << words1 << '\n';
 
	// words2 == words1
	std::vector<std::string> words2(words1.begin(), words1.end());
	std::cout << "words2: " << words2 << '\n';
 
	// words3 == words1
	std::vector<std::string> words3(words1);
	std::cout << "words3: " << words3 << '\n';
 
	// words4 is {"Mo", "Mo", "Mo", "Mo", "Mo"}
	std::vector<std::string> words4(5, "Mo");
	std::cout << "words4: " << words4 << '\n';
	
	return 0;
}
#endif
/* Test print_vector */