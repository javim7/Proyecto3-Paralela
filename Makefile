all: pgm.o	houghBase2

houghBase2:	houghBase2.cu pgm.o
	nvcc houghBase2.cu pgm.o -o houghBase2

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
