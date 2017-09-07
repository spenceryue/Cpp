#ifndef CHECK_SIGNATURE_H
#define CHECK_SIGNATURE_H

#include <type_traits>			// std::enable_if_t, std::is_convertible_v, std::true_type, std::false_type
#include <utility>				// std::declval


template <class, class, class = void>
struct check_signature : std::false_type {};

template <class Func, class Ret, class... ArgTypes>
struct check_signature<Func, Ret(ArgTypes...),
    std::enable_if_t<
        std::is_convertible_v<
            decltype(std::declval<Func>()(std::declval<ArgTypes>()...)),
		Ret>,
	void>> :
	std::true_type {};
/* Source: https://stackoverflow.com/a/25608822/3624264
   Thank you
*/

template <class Functor, class Ret, class... ArgTypes>
struct check_signature<Ret (Functor::*)(ArgTypes...), Ret(ArgTypes...),
    std::enable_if_t<
        std::is_convertible_v<
            decltype(std::declval<Functor>()(std::declval<ArgTypes>()...)),
		Ret>,
	void>> :
	std::true_type {};
/*
	I made this one myself! Celebrate:

	█▀▀▄           ▄▀▀█
	 █░░░▀▄ ▄▄▄▄▄ ▄▀░░░█
	  ▀▄░░░▀░░░░░▀░░░▄▀
	   ▐░░▄▀░░░▀▄░░▌▄▄▀▀▀▀█
	   ▌▄▄▀▀░▄░▀▀▄▄▐░░░░░░█
	▄▀▀▐▀▀░▄▄▄▄▄░▀▀▌▄▄▄░░░█
	█░░░▀▄░█░░░█░▄▀░░░░█▀▀▀
	 ▀▄░░▀░░▀▀▀░░▀░░░▄█▀
	   █░░░░░░░░░░░▄▀▄░▀▄
	   █░░░░░░░░░▄▀█  █░░█
	   █░░░░░░░░░░░█▄█░░▄▀
	   █░░░░░░░░░░░████▀
	   ▀▄▄▀▀▄▄▀▀▄▄▄█▀
*/
#endif /* CHECK_SIGNATURE_H */



/* Test check_signature */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream>			// std::cout, std::endl
	#include <thread>			// std::this_thread::sleep_for
	#include <chrono>			// std::chrono::seconds
	#include <string>			// string
	#include "type_stuff.h"		// type_name

using namespace std;
using namespace std::this_thread;
using namespace chrono;


int sleeping(int n) 
{
	cout << "Countdown:\n";
	for (int i=n; i>0; --i) {
		cout << i << endl;
		sleep_for (seconds(1));
	}
	cout << "Lift off!\n";

	return 314159265;
}

void hello()
{ cout << "hello world" << endl; }

struct callme {
	string operator() (double d) {
		cout << "lookey here: " << d << endl;
		return std::to_string(d);
	}
	using signature = decltype(&callme::operator());	// technique needed for printing type_name
};

int main(int argc, char* argv[])
{
	cout << argv[0] << " Starting..." << endl;
	cout << boolalpha << endl;
	
	cout << type_name<decltype(sleeping)>() << endl;
	cout << "+ free function with args: "	<< check_signature<decltype(sleeping), int(int)>() << endl;
	cout << "- free function with args: "	<< check_signature<decltype(sleeping), string(int)>() << "\n" << endl;
	
	cout << type_name<decltype(hello)>()	<< endl;
	cout << "+ free function no args: " 	<< check_signature<decltype(hello), void(void)>() << endl;
	cout << "- free function no args: " 	<< check_signature<decltype(hello), int(void)>() << "\n" << endl;
	
	auto hi = [] () { cout << "hi there" << endl; };
	cout << type_name<decltype(hi)>()		<< endl;
	cout << "+ lambda no args: " 			<< check_signature<decltype(hi), void(void)>() << endl;
	cout << "- lambda no args: " 			<< check_signature<decltype(hi), void(int)>() << "\n" << endl;

	cout << type_name<callme::signature>()	<< endl;
	cout << "+ functor with args: " 		<< check_signature<callme::signature, string(double)>() << endl;
	cout << "+ functor with args: " 		<< check_signature<callme, string(double)>() << endl;
	cout << "- functor with args: " 		<< check_signature<callme, char(double)>() << "\n" << endl;

	return 0;
}
#endif
/* Test check_signature */