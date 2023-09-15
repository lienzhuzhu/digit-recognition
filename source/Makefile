CXX = g++
CXXFLAGS = -std=c++11

ifdef TRAIN
	CXXFLAGS += -DTRAIN
endif

ifdef TEST
	CXXFLAGS += -DTEST
endif

INCLUDES = -Iinclude/
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

EXEC = nn

# Linker flags for SFML and OpenCV
LINKFLAGS = -lsfml-graphics -lsfml-window -lsfml-system 

ifdef DEBUG
	CXXFLAGS += -DDEBUG
	INCLUDES += -I/opt/homebrew/Cellar/opencv/4.8.0_5/include/opencv4
	LINKFLAGS += -L/opt/homebrew/Cellar/opencv/4.8.0_5/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
endif


all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LINKFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(EXEC)
