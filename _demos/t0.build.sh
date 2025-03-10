###############################################################################

# deal bad net connections
mkdir 3rd && pushd 3rd

# search `GIT_REPOSITORY`

git clone https://github.com/nvidia/cutlass.git
# git clone git@github.com:nvidia/cutlass.git
git clone https://github.com/vllm-project/FlashMLA.git
git clone https://github.com/vllm-project/flash-attention.git

pushd cutlass && git switch -d v3.8.0 && popd
pushd FlashMLA && git switch -d 575f7724b9762f265bbee5889df9c7d630801845 && popd
pushd flash-attention && git switch -d 9bfa9869829d8c593527eb34c5271d0090f7ccc9 && popd

pushd FlashMLA && git submodule update --init --depth 1 && popd
pushd flash-attention && git submodule update --init --depth 1 && popd

###############################################################################

rm -rf build/CMakeCache.txt
rm -rf build/CMakeFiles

cmake --preset demo -S. -Bbuild

# cmake --build build -t all
cmake --build build -t install

###############################################################################

pip install --no-build-isolation -e .

###############################################################################
