CXX := g++
FLAGS := 
SRC := src
INCLUDES := lib

all:
	$(CXX) $(SRC)\main.cpp -I $(INCLUDES)