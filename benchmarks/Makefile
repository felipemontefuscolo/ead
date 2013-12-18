EAD_DIR ?= ../

CXX = g++
RM  = rm -f

CPPSOURCES = $(wildcard *.cpp)
TOBJECTS = $(CPPSOURCES:.cpp=.o)
EXECS = $(CPPSOURCES:.cpp=.exe)

ADEPT_DIR=/home/felipe/libs/adept

# Flags passed to the preprocessor.
CPPFLAGS += -I$(EAD_DIR) -I $(ADEPT_DIR)
# Flags passed to the C++ compiler.
CXXFLAGS += -pg -g -Wall -Wextra
#CXXFLAGS += -O3 -march=native -mfpmath=sse -Wall -Wextra -DNDEBUG
#CXXFLAGS += -O2 -march=native -Wall -Wextra

LDFLAGS+= -L $(ADEPT_DIR)/lib -ladept

all: msg ${TOBJECTS} ${EXECS}

%.exe: %.o
	$(CXX) $(CXXFLAGS)  $< -o $@ $(LDFLAGS)

%.o: %.cpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $<  -o $@


clean:
	$(RM) *.o *.d *.exe *~
	
msg: ${EXECS}
	@echo 
	@echo all examples successfully compiled
	@echo 
