#ifndef READ_FILE_H
#define READ_FILE_H

#include <fstream> 			// std::ifstream
#include <string>			// std::stringstream
#include <iostream> 		// std::cout, std::endl

template <bool isText = false>
std::string read_file (std::string file_path)
{
	using namespace std;
	string target;
	ifstream file;

	if constexpr (isText)
		file.open(file_path, ifstream::ate);
	else
		file.open(file_path, ifstream::binary|ifstream::ate);

	if (!file) {
		cerr << "File " + file_path + " couldn't be opened. :(" << endl;
		exit(1);
	}

	int file_size = file.tellg();
	target.resize(file_size);

	file.seekg(0);
	file.rdbuf()->sgetn(&target[0], file_size);
	// file.read(target.data(), file_size);

	return target;
}

#endif /* READ_FILE_H */



/* Test read_file() */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl

using namespace std;


int main(int argc, char* argv[])
{
	string my_file = read_file<1>("read_file.h");
	
	cout << my_file << endl;
	
	return 0;
}
#endif
/* Test read_file() */