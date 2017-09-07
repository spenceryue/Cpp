#include <iostream>			// std::cout, std::cerr, std::endl
#include <fstream>			// std::ifstream, std::filebuf
#include <string>			// std::string
#include "tictoc.h"			// tictoc
#include "use_commas.h"		// use_commas

using std::ifstream;
using std::cout;
using std::cerr;
using std::endl;


#define LINE_BUFFER 256

int method1(ifstream &input, char working[LINE_BUFFER]);
int method2(ifstream &input, char working[LINE_BUFFER]);
int method3(ifstream &input, char working[LINE_BUFFER]);


int main (int argc, char** argv)
{
	using std::ios;

	/*if (argc != 2)
	{
		cerr << "line_count usage:" << endl
				  << "\tline_count <filename>" << endl;
		return -1;
	}*/

	ifstream input = (argc == 2) ?
			ifstream(argv[1], ios::binary) :
			ifstream("computing.llnl.gov.html", ios::binary);

	/*
	Candidate approaches:
		> source.getline(dest, n, delimiter)		-implemented
		> construct then read()						-implemented
		  (saves on sentry costruction)
		> rdbuf() char pointer++					-implemented
		> istream_iterator (++ ++..) (internally uses <<)
		> istringstream 
		> getline(source, dest, delimter)
		> string, for(auto& i:input) ?
	*/

	if (input.is_open())
	{
		char working[LINE_BUFFER];
		
		tictoc<method1>(input, working);

		input.clear();
		input.seekg(0); // must clear() before seekg() because eofbit is set.
		input.sync();

		tictoc<method2>(input, working);

		input.clear();
		input.seekg(0);
		input.sync();

		tictoc<method3>(input, working);

		return 0;
	}
	else
	{
		cerr << "Input file \"" << argv[1] << "\" could not be found/opened." << endl;
		return -1;
	}

}

/* Use member function of ifstream, getline() */
int method1(ifstream &input, char working[LINE_BUFFER])
{
	cout << "Method 1..." << endl;

	int count = 0;
	while (!input.eof())
	{
		input.getline(working, LINE_BUFFER);
		if (input.good())
		{
			// New line found
			count++;
		}
		else if (!input.eof()) {
			input.clear();
		}
	}
	if (input.gcount() > 0 && working[input.gcount()-1] != '\n')
		// File did not end in blank new line (count it)
		count++;
	
	cout << "Lines counted: " << use_commas(count) << endl;
	return count;
}

/* Use member function read() -- saves on sentry construction for lines less than LINE_BUFFER chars long */
int method2(ifstream &input, char working[LINE_BUFFER])
{
	cout << "Method 2..." << endl;

	int count = 0;
	while (!input.eof())
	{
		input.read(working, LINE_BUFFER);
		for (int i=0, chars_read=input.gcount();
			 i<chars_read;
			 i++)
		{
			if (working[i] == '\n')
				// New line found
				count++;
		}
	}
	if (input.gcount() > 0 && working[input.gcount()-1] != '\n')
		// File did not end in blank new line (count it)
		count++;

	cout << "Lines counted: " << use_commas(count) << endl;
	return count;
}

/* Use underlying filebuf buffer retrieved from rdbuf(), then sgetn() */
int method3(ifstream &input, char working[LINE_BUFFER])
{
	cout << "Method 3..." << endl;
	
	int count = 0,
	chars_read = 0,
	tmp = 0;

	std::filebuf *pbuf = input.rdbuf();
	// pbuf->pubseekoff(0, input.beg);	// this is what is called internally with input.seekg(0)
	while (tmp = pbuf->sgetn(working,LINE_BUFFER))
	{
		for (int i=0;
			i<tmp;
			i++)
		{
			if (working[i] == '\n')
				// New line found
				count++;
		}
		chars_read = tmp;
	}
	if (working[chars_read-1] != '\n')
		// File did not end in blank new line (count it)
		count++;

	cout << "Lines counted: " << use_commas(count) << endl;
	return count;
}