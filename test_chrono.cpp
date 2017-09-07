#include <iostream>
#include <chrono>
#include <ratio>			// std::milli
// #include "tictoc.h"

using std::cout;
using std::endl;
using std::boolalpha;
using namespace std::chrono;
using std::milli;

std::ostream cnull(0);
std::wostream wcnull(0);

steady_clock::duration func() {
	steady_clock::time_point start = steady_clock::now();

	for (int i=0; i<1000; i++) {
	cout << "system_clock" << endl;
    cout << "period numerator: " << system_clock::period::num << endl;
    cout << "period denominator: " << system_clock::period::den << endl;
    cout << "steady = " << boolalpha << system_clock::is_steady << endl << endl;

    cout << "high_resolution_clock" << endl;
    cout << "period numerator: " << high_resolution_clock::period::num << endl;
    cout << "period denominator: " << high_resolution_clock::period::den << endl;
    cout << "steady = " << boolalpha << high_resolution_clock::is_steady << endl << endl;

    cout << "steady_clock" << endl;
    cout << "period numerator: " << steady_clock::period::num << endl;
    cout << "period denominator: " << steady_clock::period::den << endl;
    cout << "steady = " << boolalpha << steady_clock::is_steady << endl << endl;
}
	steady_clock::time_point end = steady_clock::now();
	return end - start;
}

int main(){
	
	
    // steady_clock::duration retValue = tictoc(func);
    steady_clock::duration retValue = func();
	

	/*
	implicit conversion constructor allowed because destination representation
	type is floating point (hence always wide enough)
	see here for more details:
	http://www.cplusplus.com/reference/chrono/duration/duration/
	*/
	// duration<double, milli> elapsed_ms = end - start; // decimal milliseconds
	duration<double, milli> elapsed_ms = retValue; // decimal milliseconds

	// milliseconds elapsed_ms_i = duration_cast<milliseconds>(end - start); // integral milliseconds
	milliseconds elapsed_ms_i = duration_cast<milliseconds>(retValue); // integral milliseconds

	cout << "Floating milliseconds: " << elapsed_ms.count() << endl;
	cout << "Integral milliseconds: " << elapsed_ms_i.count() << endl;
	cout << "Integral nanoseconds: " << (retValue).count() << endl;

    return 0;    
}