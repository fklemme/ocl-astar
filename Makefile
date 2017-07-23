BOOSTDIR := $(shell test -d boost && echo local)

CXXFLAGS += -std=c++14 \
            -O3 \
            -Wno-ignored-attributes
            #-Wall -Wextra
            #-Wno-unknown-pragmas
LDFLAGS  += -lOpenCL

ifeq ($(BOOSTDIR), local)
    CXXFLAGS += -Iboost
endif

HEADERS := $(wildcard src/*.h)
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(addprefix obj/,$(notdir $(SOURCES:.cpp=.o)))

.PHONY: run
run: ocl-astar
	./$<

.PHONY: display
display:
	for pfm in $$(find . -type f -name "*.pfm"); do \
	    display "$$pfm" & \
	done

ocl-astar: obj $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

obj:
	mkdir obj

obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf obj

boost:
	curl -L https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2 | tar xj
	mv boost_1_64_0 boost

# Clang Format - Settings in file .clang-format
.PHONY: format
format:
	clang-format -i -style=file $(HEADERS) $(SOURCES)
