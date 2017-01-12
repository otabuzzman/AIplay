ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

shaffer: shaffer.c
	$(CXX) -Wall -o $@ $<
