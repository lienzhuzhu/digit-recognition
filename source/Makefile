# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11

ifdef TRAIN
	CXXFLAGS += -DTRAIN
endif

ifdef TEST
	CXXFLAGS += -DTEST
endif

# Include directories
INCLUDES = -Iinclude/
INCLUDES += -I/opt/homebrew/include/

# Source files
SOURCES = $(wildcard *.cpp)

# Executable name
EXEC = nn

# Linker flag for SFML in homebrew
LINKFLAGS = -L/opt/homebrew/lib/ -lsfml-graphics -lsfml-window -lsfml-system

ifdef DEBUG
	CXXFLAGS += -DDEBUG
	INCLUDES += -I/opt/homebrew/Cellar/opencv/4.8.0_5/include/opencv4
	LINKFLAGS += -L/opt/homebrew/Cellar/opencv/4.8.0_5/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
endif


# Rules
all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(EXEC) $(LINKFLAGS)

clean:
	rm -f $(EXEC)

