EAD_DIR ?= ../
ADEPT_DIR ?=/home/felipe/libs/adept
FADBAD_DIR ?= /home/felipe/libs/FADBAD++

CXX = g++
RM  = rm -f

CPPSOURCES = $(wildcard *.cpp)
TOBJECTS = $(CPPSOURCES:.cpp=.o)
EXECS = $(CPPSOURCES:.cpp=.exe)

ifneq "" "$(wildcard ${ADEPT_DIR})"
CPPFLAGS += -I $(ADEPT_DIR) -DHAS_ADEPT
LDFLAGS+= -L $(ADEPT_DIR)/lib -ladept -DHAS_ADEPT
endif

ifneq "" "$(wildcard ${FADBAD_DIR})"
CPPFLAGS += -I $(FADBAD_DIR) -DHAS_FADBAD
LDFLAGS+= -L $(FADBAD_DIR)/lib -DHAS_FADBAD
endif

# Flags passed to the preprocessor.
CPPFLAGS += -I$(EAD_DIR)
# Flags passed to the C++ compiler.
#CXXFLAGS += -O2 -g -Wall -Wextra -fno-inline
#CXXFLAGS += -O3 -march=native -ftree-vectorize -msse2 -ftree-vectorizer-verbose=5 -Wall -Wextra -DNDEBUG
CXXFLAGS += -O3 -march=native -ftree-vectorize -msse2 -Wall -Wextra -DNDEBUG
#CXXFLAGS += -O2 -march=native -Wall -Wextra

# valgrind --tool=callgrind  --simulate-cache=yes --collect-jumps=yes  --dump-instr=yes --trace-jump=yes  --branch-sim=yes



all: msg ${TOBJECTS} ${EXECS}

%.exe: %.o
	$(CXX) $(CXXFLAGS)  $< -o $@ $(LDFLAGS)

%.o: %.cpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $<  -o $@


clean:
	$(RM) *.o *.d *.exe *~ callgrind.o*
	
msg: ${EXECS}
	@echo 
	@echo all examples successfully compiled
	@echo 
