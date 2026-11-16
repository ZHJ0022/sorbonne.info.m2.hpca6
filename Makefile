NVCC = nvcc
NVCCFLAGS = -std=c++17 -Iinclude
LDLIBS = -lcurand
TARGET = question1

SOURCES = src/question1/main.cu src/question1/simulation.cu src/question1/output.cpp
HEADERS = include/question1.cuh include/hull_white.cuh

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)
