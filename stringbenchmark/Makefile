CXXFLAGS = -O3 -std=c++17
TARGETS  = index.exe \
		   cindex.exe \
		   cpointers.exe \
		   iterators.exe \
		   stringat.exe \
		   const_iterators.exe \
		   for_each.exe \
		   for_reference_range_loop.exe \
		   for_const_reference_range_loop.exe
VPATH = bin/


all: $(TARGETS)

clean:
	rm bin/ -rf

refresh:
	touch hugestring.dll

index.exe: index.cpp hugestring.dll
	g++ $(CXXFLAGS) index.cpp -o bin/index.exe bin/hugestring.dll

cindex.exe: cindex.cpp hugestring.dll
	g++ $(CXXFLAGS) cindex.cpp -o bin/cindex.exe bin/hugestring.dll

cpointers.exe: cpointers.cpp hugestring.dll
	g++ $(CXXFLAGS) cpointers.cpp -o bin/cpointers.exe bin/hugestring.dll

iterators.exe: iterators.cpp hugestring.dll
	g++ $(CXXFLAGS) iterators.cpp -o bin/iterators.exe bin/hugestring.dll

const_iterators.exe: const_iterators.cpp hugestring.dll
	g++ $(CXXFLAGS) const_iterators.cpp -o bin/const_iterators.exe bin/hugestring.dll

for_each.exe: for_each.cpp hugestring.dll
	g++ $(CXXFLAGS) for_each.cpp -o bin/for_each.exe bin/hugestring.dll

for_reference_range_loop.exe: for_reference_range_loop.cpp hugestring.dll
	g++ $(CXXFLAGS) for_reference_range_loop.cpp -o bin/for_reference_range_loop.exe bin/hugestring.dll

for_const_reference_range_loop.exe: for_const_reference_range_loop.cpp hugestring.dll
	g++ $(CXXFLAGS) for_const_reference_range_loop.cpp -o bin/for_const_reference_range_loop.exe bin/hugestring.dll

stringat.exe: stringat.cpp hugestring.dll
	g++ $(CXXFLAGS) stringat.cpp -o bin/stringat.exe bin/hugestring.dll

hugestring.dll: generatestring.exe
	# .blob extension means nothing at all
	./bin/generatestring.exe > bin/hugestring.blob
	# -x flag says to read the .blob as a c++ file (MUST precede the file in argument list)
	# -shared, and -fpic (position-independent code) make it a shared library
	# .dll extension doesn't matter at all (could have been .so and work on Windows)
	g++ -x c++ bin/hugestring.blob -o bin/hugestring.dll -shared -fpic
	rm bin/hugestring.blob

generatestring.exe: generatestring.cpp | bin/
	g++ generatestring.cpp -o bin/generatestring.exe
	
bin/:
	[ ! -d bin/ ] && mkdir bin/ || true
