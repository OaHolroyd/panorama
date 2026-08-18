# Raytracing and tile-generation command-line executables.
PANORAMA_EXE := panorama
TILE_GEN_EXE := panorama-tile-gen

# Keep each executable's implementation separate. Future file-format code can
# live in shared/ and be linked into both without either depending on the
# other's entry point or application-specific implementation.
SRC_DIR := src
RAYTRACE_SRC_DIR := $(SRC_DIR)/raytracing
TILE_GEN_SRC_DIR := $(SRC_DIR)/tile-gen
SHARED_SRC_DIR := $(SRC_DIR)/shared
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
CPPFLAGS := $(GDAL_CFLAGS)
RAYTRACE_INCLUDES := -I$(RAYTRACE_SRC_DIR) -I$(SHARED_SRC_DIR)
TILE_GEN_INCLUDES := -I$(TILE_GEN_SRC_DIR) -I$(SHARED_SRC_DIR)
SHARED_INCLUDES := -I$(SHARED_SRC_DIR)
WARNINGS := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
COMMON_FLAGS := -std=c++20 -fobjc-arc -MMD -MP
# ImageIO/CoreGraphics encode diagnostic and rendered images as PNG files.
FRAMEWORKS := -framework Foundation -framework Metal -framework CoreGraphics -framework ImageIO
LDLIBS := $(GDAL_LIBS)

# Select with `make DEBUG=1`; release is the default.
DEBUG ?= 0
ifeq ($(DEBUG),1)
  BUILD := debug
  OPT_FLAGS := -O0 -g
  # Expensive host-frontier invariant checks are compiled only in debug builds.
  CPPFLAGS += -DPANORAMA_DEBUG_VALIDATION=1
else
  BUILD := release
  OPT_FLAGS := -O3
endif

# Separate configurations so `make DEBUG=1` cannot reuse release objects.
OBJ_DIR := $(OBJ_ROOT)/$(BUILD)

# Each executable has its own source and object directory. Shared translation
# units are compiled once per build configuration and linked into both.
PANORAMA_SRC := $(wildcard $(RAYTRACE_SRC_DIR)/*.mm)
PANORAMA_OBJ := $(patsubst $(RAYTRACE_SRC_DIR)/%.mm,$(OBJ_DIR)/raytracing/%.o,$(PANORAMA_SRC))
TILE_GEN_SRC := $(wildcard $(TILE_GEN_SRC_DIR)/*.mm)
TILE_GEN_OBJ := $(patsubst $(TILE_GEN_SRC_DIR)/%.mm,$(OBJ_DIR)/tile-gen/%.o,$(TILE_GEN_SRC))
SHARED_SRC := $(wildcard $(SHARED_SRC_DIR)/*.mm)
SHARED_OBJ := $(patsubst $(SHARED_SRC_DIR)/%.mm,$(OBJ_DIR)/shared/%.o,$(SHARED_SRC))
METAL_SRC := $(wildcard $(RAYTRACE_SRC_DIR)/*.metal)
METAL_AIR := $(patsubst $(RAYTRACE_SRC_DIR)/%.metal,$(OBJ_DIR)/raytracing/%.air,$(METAL_SRC))
METAL_LIB := $(OBJ_DIR)/raytracing/panorama.metallib
# Tell the host where this configuration's generated metallib is located.
RAYTRACE_DEFINES := -DPANORAMA_METALLIB_PATH=\"$(METAL_LIB)\"
# Compiler-generated header dependencies for the Objective-C++ sources.
DEPS := $(PANORAMA_OBJ:.o=.d) $(TILE_GEN_OBJ:.o=.d) $(SHARED_OBJ:.o=.d)

.PHONY: all clean rebuild compile_commands FORCE

all: $(PANORAMA_EXE) $(TILE_GEN_EXE)

# The executable name is shared by configurations, so relink it to the
# configuration requested by this invocation even if the other build was newer.
$(PANORAMA_EXE): FORCE $(PANORAMA_OBJ) $(SHARED_OBJ) $(METAL_LIB)
	@printf 'Linking %s\n' '$@'
	$(CXX) $(OPT_FLAGS) -o $@ $(PANORAMA_OBJ) $(SHARED_OBJ) $(FRAMEWORKS) $(LDLIBS)

# The tile generator does not use a metallib yet. It links Metal because the
# custom-tile writer uses Metal I/O compression contexts.
$(TILE_GEN_EXE): FORCE $(TILE_GEN_OBJ) $(SHARED_OBJ)
	@printf 'Linking %s\n' '$@'
	$(CXX) $(OPT_FLAGS) -o $@ $(TILE_GEN_OBJ) $(SHARED_OBJ) -framework Foundation -framework Metal $(LDLIBS)

# Compile the raytracer and tile generator into distinct object directories so
# their respective main.mm files cannot collide.
$(OBJ_DIR)/raytracing/%.o: $(RAYTRACE_SRC_DIR)/%.mm | $(OBJ_DIR)/raytracing
	@printf 'Compiling %s\n' '$@'
	$(CXX) $(RAYTRACE_INCLUDES) $(CPPFLAGS) $(RAYTRACE_DEFINES) $(COMMON_FLAGS) $(WARNINGS) $(OPT_FLAGS) -c -o $@ $<

$(OBJ_DIR)/tile-gen/%.o: $(TILE_GEN_SRC_DIR)/%.mm | $(OBJ_DIR)/tile-gen
	@printf 'Compiling %s\n' '$@'
	$(CXX) $(TILE_GEN_INCLUDES) $(CPPFLAGS) $(COMMON_FLAGS) $(WARNINGS) $(OPT_FLAGS) -c -o $@ $<

$(OBJ_DIR)/shared/%.o: $(SHARED_SRC_DIR)/%.mm | $(OBJ_DIR)/shared
	@printf 'Compiling %s\n' '$@'
	$(CXX) $(SHARED_INCLUDES) $(CPPFLAGS) $(COMMON_FLAGS) $(WARNINGS) $(OPT_FLAGS) -c -o $@ $<

# Metal source compiles to AIR (Apple Intermediate Representation) before the
# metallib tool packages every AIR file into one library.
$(OBJ_DIR)/raytracing/%.air: $(RAYTRACE_SRC_DIR)/%.metal | $(OBJ_DIR)/raytracing
	@printf 'Compiling %s\n' '$@'
	$(METAL) -c -o $@ $<

$(METAL_LIB): $(METAL_AIR)
	@printf 'Creating %s\n' '$@'
	$(METALLIB) -o $@ $<

$(OBJ_DIR)/raytracing $(OBJ_DIR)/tile-gen $(OBJ_DIR)/shared:
	mkdir -p $@

# Generate the compilation database consumed by clangd/objc-clangd. Bear
# records the compiler invocations that Make actually executes, so this does
# not duplicate compiler flags or source discovery in a hand-written JSON
# recipe. `-B` deliberately rebuilds the executable: there must be real
# compiler processes for Bear to observe.
compile_commands:
	rm -f compile_commands.json
	bear --output compile_commands.json -- $(MAKE) --no-print-directory -B all

rebuild: clean all

clean:
	rm -rf $(PANORAMA_EXE) $(TILE_GEN_EXE) $(OBJ_ROOT) $(COMPILE_DB)

# A phony prerequisite makes the shared executable relink when switching
# between debug and release object directories.
FORCE:

# Missing dependency files are harmless on the first build.
-include $(DEPS)
