#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    """
    Build the poker_core C++ module directly using CMake and pybind11.
    """
    print("Building poker_core module...")
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    
    # Use the current directory for in-source build
    build_dir = script_dir
    
    # Clean CMake cache and make sure we build from scratch
    for clean_file in ["CMakeCache.txt"]:
        clean_path = build_dir / clean_file
        if clean_path.exists():
            print(f"Removing {clean_file} to ensure clean build")
            clean_path.unlink()
            
    # Remove any existing pybind11 or googletest build directories
    for dir_to_clean in ["_deps/pybind11-subbuild", "_deps/googletest-subbuild"]:
        clean_path = build_dir / dir_to_clean
        if clean_path.exists() and clean_path.is_dir():
            print(f"Removing {dir_to_clean} to ensure clean build")
            import shutil
            shutil.rmtree(clean_path, ignore_errors=True)
    
    # Get Python executable and include paths
    python_executable = sys.executable
    
    # Run CMake
    cmake_cmd = [
        "cmake", 
        ".",
        f"-DPYTHON_EXECUTABLE={python_executable}",
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    
    print(f"Running CMake: {' '.join(cmake_cmd)}")
    result = subprocess.run(cmake_cmd, check=False)
    if result.returncode != 0:
        print(f"CMake configuration failed with code {result.returncode}")
        return result.returncode
    
    # Run make
    num_cores = os.cpu_count() or 2
    make_cmd = ["make", f"-j{num_cores}"]
    
    print(f"Running Make: {' '.join(make_cmd)}")
    result = subprocess.run(make_cmd, check=False)
    if result.returncode != 0:
        print(f"Make failed with code {result.returncode}")
        return result.returncode
    
    print("Build successful!")
    
    # List the built library files
    print("\nBuilt Files:")
    
    # Track the module location
    module_found = False
    
    for path in build_dir.glob("**/*.so"):
        rel_path = path.relative_to(script_dir)
        print(f" - {rel_path}")
        
        # If this is the poker_core module, make sure it's accessible (but don't try to copy to itself)
        if "poker_core" in path.name:
            # Check if the path is different from our destination
            dest_path = script_dir / path.name
            if path.resolve() != dest_path.resolve():
                print(f"Copying {rel_path} to {path.name} for easier importing")
                import shutil
                shutil.copy(path, dest_path)
            else:
                print(f"Module {path.name} is already in the correct location")
            module_found = True
    
    if not module_found:
        print("Warning: poker_core module was not found in build output!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())