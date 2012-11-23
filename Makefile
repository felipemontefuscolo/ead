OPT=-O3

tests: tests.o
	g++ ${OPT} -Wall -Wextra -pedantic tests.o -o tests
	rm tests.o

tests.o: tests.cpp
	g++ ${OPT} -c -Wall -Wextra -pedantic tests.cpp -o tests.o

clean:
	rm tests.o
	rm tests


