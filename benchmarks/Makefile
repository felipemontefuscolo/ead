EAD_DIR ?= ../

CXX = g++
RM  = rm -f

CPPSOURCES = $(wildcard *.cpp)
TOBJECTS = $(CPPSOURCES:.cpp=.o)
EXECS = $(CPPSOURCES:.cpp=.exe)

# Flags passed to the preprocessor.
CPPFLAGS += -I$(EAD_DIR)
# Flags passed to the C++ compiler.
CXXFLAGS += -pg -Wall -Wextra
#CXXFLAGS += -O3 -march=native -mfpmath=sse -Wall -Wextra
#CXXFLAGS += -O2 -march=native -Wall -Wextra

all: msg ${TOBJECTS} ${EXECS}

%.exe: %.o
	$(CXX) $(CXXFLAGS) $< -o $@

%.o: %.cpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< $(LDFLAGS) -o $@


clean:
	$(RM) *.o *.d *.exe *~
	
msg: ${EXECS}
	@echo 
	@echo all examples successfully compiled
	@echo 
