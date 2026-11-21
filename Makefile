NVCC = nvcc
NVCCFLAGS = -std=c++17 -Iinclude

Q1_SOURCES = src/question1/main.cu src/question1/simulation.cu src/question1/output.cpp
Q1_HEADERS = include/question1.cuh include/hull_white.cuh
Q1_LDLIBS = -lcurand

Q2_SOURCES = src/question2/main.cu src/question2/recover_theta.cu src/question2/io.cpp
Q2_HEADERS = include/question2.cuh include/question1.cuh include/hull_white.cuh

.PHONY: all clean

all: question1 question2

question1: $(Q1_SOURCES) $(Q1_HEADERS)
	$(NVCC) $(NVCCFLAGS) $(Q1_SOURCES) -o $@ $(Q1_LDLIBS)

question2: $(Q2_SOURCES) $(Q2_HEADERS)
	$(NVCC) $(NVCCFLAGS) $(Q2_SOURCES) -o $@

clean:
	rm -f question1 question2 question1.csv
	rm -f results/question1.png results/question1_summary.txt results/question2_theta.csv
	rm -rf build/
