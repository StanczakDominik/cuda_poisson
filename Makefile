NVCC = nvcc
NVCC_FLAGS = -arch=sm_20 -rdc=true  --use_fast_math -lcufft -g -G

all: main.out

main.out: main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe *.out

run:
	make
	./main.out
