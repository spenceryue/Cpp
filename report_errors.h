#ifndef REPORT_ERRORS_H
#define REPORT_ERRORS_H
#include <iostream> 			// std::cerr, std::endl
#include <string> 				// std::string
#include <exception> 			// std::exception
#include "faces.h"

#define throw_err( msg ) report::error( (msg), __FILE__, __LINE__)
#define warn_err( msg ) report::error<false>( (msg), __FILE__, __LINE__)

namespace report {
	template <bool except = true>
	constexpr bool error (const std::string& msg, const std::string& file, const int line)
	{
		using namespace std;

		cout << flush;
		
		cerr
			<< "\n" flip_table "\n"
			<< "Error in \"" << file << "\" just before line: " << to_string(line) << "\n"
			<< msg + "\n"
			<< endl;

		if constexpr (except)
			throw exception();

		return true;
	}
}
#endif /* REPORT_ERRORS_H */



/* Test report_errors */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl

using namespace std;


int main(int argc, char* argv[])
{
	try {
	cout << "Hello World!" << endl;

	warn_err("Oh my... but,,, not fatal. continue...");
	cout << "Hello World,#!@#$%^&*()!" << endl;
	throw_err("Oh no! Oh dear.");

	} catch (...) {return 1;}

	return 0;
}
#endif
/* Test report_errors */