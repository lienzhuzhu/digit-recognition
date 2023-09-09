# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11

# Include directories
INCLUDES = -Iinclude/ -I/opt/homebrew/Cellar/opencv/4.8.0_5/include/opencv4

# Source files
SOURCES = $(wildcard *.cpp)

# Executable name
EXEC = nn

# Linker flags for SFML and OpenCV
LINKFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -L/opt/homebrew/Cellar/opencv/4.8.0_5/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

# Rules
all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(EXEC) $(LINKFLAGS)

clean:
	rm -f $(EXEC)

