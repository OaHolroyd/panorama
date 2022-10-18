# ============================================================================ #
#   VARIABLE DEFINITIONS                                                       #
# ============================================================================ #
# compiler/linker
CC=gcc-12
LD=$(CC)

# flags
# CFLAGS=-O0 -g -Wall -Wextra -DDEBUG -fbounds-check \
#        -fsanitize=address -fsanitize=bounds -fsanitize=bounds-strict
CFLAGS=-O0 -g -Wall -Wextra -pedantic -Wno-unused-parameter -Wshadow \
       -Waggregate-return -Wbad-function-cast -Wcast-align -Wcast-qual \
       -Wfloat-equal -Wformat=2 -Wlogical-op -Wmissing-include-dirs \
       -Wnested-externs -Wpointer-arith -Wconversion -Wno-sign-conversion \
       -Wredundant-decls -Wsequence-point -Wstrict-prototypes -Wswitch -Wundef \
       -Wunused-but-set-parameter -Wwrite-strings
LDFLAGS=

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
	$(LD) $(CFLAGS) -o $(EXE) $(OBJ) $(LDFLAGS)

# compile/assemble object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c Makefile
	@printf "`tput bold``tput setaf 6`Compiling %s`tput sgr0`\n" $@
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# include dependency information
-include $(DEPS)

# force rebuild of all files
.PHONY: all
all: clean $(EXE)

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
