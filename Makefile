CXXFLAGS += -std=c++14 \
            -O3 \
            -Wall -Wextra
LDFLAGS  += -OpenCL

HEADERS := $(wildcard src/*.h)
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(addprefix obj/,$(notdir $(SOURCES:.cpp=.o)))

.PHONY: all
all: ocl-astar

ocl-astar: obj $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

obj:
	mkdir obj

obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf obj

# Clang Format - Settings in file .clang-format
.PHONY: format
format:
	clang-format -i -style=file $(HEADERS) $(SOURCES)

# Clang Tidy
.PHONY: tidy
tidy:
	clang-tidy -checks=cppcoreguidelines-*,modernize-*,readability-*,-readability-braces-around-statements,-cppcoreguidelines-pro-bounds-array-to-pointer-decay \
	    -header-filter=src/ $(SOURCES) -- $(CXXFLAGS)
