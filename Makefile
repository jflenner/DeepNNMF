CC = /usr/lib/llvm-3.8/bin/clang++ 
CFLAGS = -std=c++11 -pthread -O3
INCLUDE = -IMachineLearning -I/usr/include -I/usr/include/eigen3/Eigen
LIBPATH = -L/usr/lib/x86_64-linux-gnu
LIBRARIES = -lpthread 
BINARIES = deepNMF

all: $(BINARIES)

deepNMF: DeepNNMF/main.cpp
	$(CC) $(CFLAGS) -o deepNMF DeepNNMF/main.cpp $(INCLUDE) $(LIBPATH) $(LIBRARIES)

clean:
	rm $(BINARIES)
