CXXFLAGS += -std=c++14 \
            -O3 \
            -Wall -Wextra \
			-Wno-unknown-pragmas
LDFLAGS  += -lOpenCL

HEADERS := $(wildcard src/*.h)
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(addprefix obj/,$(notdir $(SOURCES:.cpp=.o)))

.PHONY: run
run: ocl-astar
	./$<

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
