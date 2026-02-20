cd $PWD/../..
cd sglang/python
pip install -e . --break-system-packages
pip install -U transformers --break-system-packages
cd ../..
pip install flashinfer_jit_cache-0.6.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl --break-system-packages
cd projects/Intern-S1
pip install -r requirements_slime.txt --break-system-packages