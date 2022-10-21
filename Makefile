# ============================================================================ #
#   VARIABLE DEFINITIONS                                                       #
# ============================================================================ #
# compiler/linker
CC=gcc-12
LD=$(CC)

# flags
WARNINGS=-Wall -Wextra -pedantic -Wno-unused-parameter -Wshadow \
         -Waggregate-return -Wbad-function-cast -Wcast-align -Wcast-qual \
         -Wfloat-equal -Wformat=2 -Wlogical-op -Wmissing-include-dirs \
         -Wnested-externs -Wpointer-arith -Wconversion -Wno-sign-conversion \
         -Wredundant-decls -Wsequence-point -Wstrict-prototypes -Wswitch -Wundef \
         -Wunused-but-set-parameter -Wwrite-strings
DEBUG=-O0 -DDEBUG -fbounds-check \
      -fsanitize=address -fsanitize=bounds -fsanitize=bounds-strict
CFLAGS=-Ofast $(WARNINGS)
LDLIBS=-lpng

# executable
EXE=panorama

# directories
SRC_DIR=./src
OBJ_DIR=./obj
OUT_DIR=./out

# files
SRC=$(wildcard $(SRC_DIR)/*.c)
OBJ=$(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.c=.o)))
DEPS=$(patsubst %.o,%.d,$(OBJ)) # dependency files


# ============================================================================ #
#   RULES                                                                      #
# ============================================================================ #
# link objects into single binary
$(EXE): directories $(OBJ)
	@printf "`tput bold``tput setaf 2`Linking`tput sgr0`\n"
	$(LD) $(CFLAGS) $(LDFLAGS) -o $(EXE) $(OBJ) $(LDLIBS)

# compile/assemble object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c Makefile
	@printf "`tput bold``tput setaf 6`Compiling %s`tput sgr0`\n" $@
	$(CC) $(CFLAGS) $(LDFLAGS) -MMD -MP -c -o $@ $< $(LDLIBS)

# include dependency information
-include $(DEPS)

# force rebuild of all files
.PHONY: all
all: clean $(EXE)

# forces a debug build
.PHONY: debug
debug: CFLAGS=$(DEBUG) $(WARNINGS)
debug: all

# create required directories
.PHONY: directories
directories:
	@printf "`tput bold``tput setaf 3`Creating directories`tput sgr0`\n"
	mkdir -p $(OBJ_DIR) $(OUT_DIR)

# purge build files and executable
.PHONY: clean
clean:
	@printf "`tput bold``tput setaf 1`Cleaning`tput sgr0`\n"
	rm -rf $(OBJ_DIR)/* $(EXE)
