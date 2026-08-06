# The command-line executable produced by the default target.
EXE := panorama
COMPILE_DB := compile_commands.json

# Keep source files flat for now; all generated files stay below obj/.
SRC_DIR := src
OBJ_ROOT := obj

# Objective-C++ host compiler and the two Metal shader-toolchain stages.
CXX := clang++
METAL := xcrun --sdk macosx metal
METALLIB := xcrun --sdk macosx metallib
GDAL_CONFIG := gdal-config
# GDAL supplies GeoTIFF loading and EPSG/PROJ-backed coordinate transforms.
# Mark its headers as system headers so third-party warnings do not obscure ours.
GDAL_INCLUDE_DIR := $(patsubst -I%,%,$(shell $(GDAL_CONFIG) --cflags))
GDAL_CFLAGS := -isystem $(GDAL_INCLUDE_DIR)
GDAL_LIBS := $(shell $(GDAL_CONFIG) --libs)

# Host compiler options.  -MMD/-MP generate makefile dependency files next to
# each object, and ARC is required by the Objective-C++ Metal host code.
CPPFLAGS := -I$(SRC_DIR)
CPPFLAGS += $(GDAL_CFLAGS)
WARNINGS := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
COMMON_FLAGS := -std=c++20 -fobjc-arc -MMD -MP
FRAMEWORKS := -framework Foundation -framework Metal
LDLIBS := $(GDAL_LIBS)

# Select with `make DEBUG=1`; release is the default.
DEBUG ?= 0
ifeq ($(DEBUG),1)
  BUILD := debug
  OPT_FLAGS := -O0 -g
else
  BUILD := release
  OPT_FLAGS := -O3
endif

# Separate configurations so `make DEBUG=1` cannot reuse release objects.
OBJ_DIR := $(OBJ_ROOT)/$(BUILD)

# Discover every host and shader source in src/.  Adding a .mm or .metal file
# therefore needs no Makefile edit.  All shaders share one runtime library.
HOST_SRC := $(wildcard $(SRC_DIR)/*.mm)
HOST_OBJ := $(patsubst $(SRC_DIR)/%.mm,$(OBJ_DIR)/%.o,$(HOST_SRC))
METAL_SRC := $(wildcard $(SRC_DIR)/*.metal)
METAL_AIR := $(patsubst $(SRC_DIR)/%.metal,$(OBJ_DIR)/%.air,$(METAL_SRC))
METAL_LIB := $(OBJ_DIR)/panorama.metallib
# Tell the host where this configuration's generated metallib is located.
CPPFLAGS += -DPANORAMA_METALLIB_PATH=\"$(METAL_LIB)\"
# Compiler-generated header dependencies for the Objective-C++ sources.
DEPS := $(HOST_OBJ:.o=.d)

.PHONY: all clean rebuild run compile_commands FORCE

all: $(EXE) $(COMPILE_DB)

# The executable name is shared by configurations, so relink it to the
# configuration requested by this invocation even if the other build was newer.
$(EXE): FORCE $(HOST_OBJ) $(METAL_LIB)
	@printf 'Linking %s\n' '$@'
	$(CXX) $(OPT_FLAGS) -o $@ $(HOST_OBJ) $(FRAMEWORKS) $(LDLIBS)

# Compile one Objective-C++ source file into its matching object file.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm | $(OBJ_DIR)
	@printf 'Compiling %s\n' '$@'
	$(CXX) $(CPPFLAGS) $(COMMON_FLAGS) $(WARNINGS) $(OPT_FLAGS) -c -o $@ $<

# Metal source compiles to AIR (Apple Intermediate Representation) before the
# metallib tool packages every AIR file into one library.
$(OBJ_DIR)/%.air: $(SRC_DIR)/%.metal | $(OBJ_DIR)
	@printf 'Compiling %s\n' '$@'
	$(METAL) -c -o $@ $<

$(METAL_LIB): $(METAL_AIR)
	@printf 'Creating %s\n' '$@'
	$(METALLIB) -o $@ $<

$(OBJ_DIR):
	mkdir -p $@

# Generate the compilation database consumed by clangd/objc-clangd.  This is
# deliberately a separate target from linking: it describes each source-file
# compilation, including the GDAL header directory, rather than link commands.
compile_commands: $(COMPILE_DB)

$(COMPILE_DB): FORCE $(HOST_SRC) Makefile
	@{ \
		printf '[\n'; \
		first=1; \
		for source in $(HOST_SRC); do \
			if [ $$first -eq 0 ]; then printf ',\n'; fi; \
			printf '  {"directory":"%s","arguments":["%s","-Isrc",' '$(CURDIR)' '$(CXX)'; \
			printf '"-isystem","%s","-std=c++20","-fobjc-arc",' '$(GDAL_INCLUDE_DIR)'; \
			printf '"-x","objective-c++","-c","%s"],"file":"%s"}' "$$source" "$$source"; \
			first=0; \
		done; \
		printf '\n]\n'; \
	} > $@
# Convenience target for the default input size used by the dummy kernel.
run: $(EXE)
	./$(EXE)

rebuild: clean all

clean:
	rm -rf $(EXE) $(OBJ_ROOT) $(COMPILE_DB)

# A phony prerequisite makes the shared executable relink when switching
# between debug and release object directories.
FORCE:

# Missing dependency files are harmless on the first build.
-include $(DEPS)
