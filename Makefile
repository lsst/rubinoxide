TARGET_DIR = pylib
WHEEL_DIR = target/wheels

UNAME_S := $(shell uname -s)

# Because we link againstconda's openblas we need to tell conda test where
# to find the library when it runs. This is a bit different on mac and
# linux as the mac version of rust MUST run with the system libc++. This
# means it must be first in the library search path.
ifeq ($(UNAME_S),Darwin)
	LIB_PREFIX := DYLD_LIBRARY_PATH=/usr/lib:$(CONDA_PREFIX)/lib:${DYLD_LIBRARY_PATH}
else
	LIB_PREFIX := LD_LIBRARY_PATH=/usr/lib:$(CONDA_PREFIX)/lib:${LD_LIBRARY_PATH}
endif

all: clean _build install test

clean:
	@echo "Cleaning existing code..."
	rm -rf $(TARGET_DIR)/*
	rm -rf $(WHEEL_DIR)/*.whl

# Turning manylinux off for default builds as it can cause bundling issues
# if we end up wanting it for pip packages or something it should be added
# as a separate make target.
_build: clean
	@echo "Building wheel..."
	maturin build -r --manylinux off

install: _build
	@echo "Adding wheel to eups environment..."
	mkdir -p $(TARGET_DIR)
	pip install -qqq $(WHEEL_DIR)/*.whl --target=$(TARGET_DIR)

test: install
	@echo "Running tests..."
	$(LIB_PREFIX) cargo test
	pytest tests

.PHONY: all clean _build install test
