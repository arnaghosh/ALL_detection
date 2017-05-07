all:

tif2jpg: tif2jpg.cpp
	g++ -g $^ -std=c++0x -o $@ -Wall `pkg-config opencv --cflags --libs`

dataAugment: dataAugmentFromOriginal.cpp
	g++ -g $^ -std=c++0x -o $@ -Wall `pkg-config opencv --cflags --libs`
