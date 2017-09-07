#ifndef BASENAME_H
#define BASENAME_H

#include <iostream>				// std::cout, std::endl;
#include <string_view>			// std::string_view


inline auto filename (std::string_view filepath)
{
	auto slash = filepath.find_last_of("\\/");
	if (slash == filepath.npos)
		return filepath;

	return filepath.substr(slash + 1);
}

inline auto basename (std::string_view filepath)
{
	filepath = filename(filepath);
	auto dot = filepath.find_last_of('.');
	if (dot == filepath.npos)
		return filepath;
	
	return filepath.substr(0, dot);
}

inline auto dirname (std::string_view filepath)
{
	auto slash = filepath.find_last_of("\\/");
	if (slash == filepath.npos)
		return filepath;

	return filepath.substr(0, slash + 1);
}

inline void startup_msg (std::string_view path)
{
	std::cout << "\"" << path << "\" starting..." << "\n" << std::endl;
}

#endif /* BASENAME_H */



/* Test basename() */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <string>	 		// std::string

using namespace std;


int main(int argc, char* argv[])
{
	startup_msg(argv[0]);
	cout << "filename:\t" << filename(argv[0] + ".exe"s) << endl;
	cout << "basename:\t" << basename(argv[0] + ".exe"s) << endl;
	cout << "dirname:\t" << dirname(argv[0] + ".exe"s) << endl;

	return 0;
}
#endif
/* Test basename() */