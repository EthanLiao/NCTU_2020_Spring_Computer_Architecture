CC=g++
LINKER_DIRS=-L/usr/local/cuda/lib
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`

all: main

main: main.o device.o 
	$(CC) main.o device.o -o main $(LINKER_DIRS) $(OCELOT)

main.o: main.cu
	$(NVCC) main.cu -c -I . 

device.o: device.cu
	$(NVCC) -c device.cu -arch=sm_$(CUDA_ARCHITECTURE) -I .

clean:
	rm -f main.o device.o main kernel-times.json
