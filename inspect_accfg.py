
import inspect
from accfg import AccFG

print("AccFG __init__ signature:")
print(inspect.signature(AccFG.__init__))

print("\nAccFG.run signature:")
print(inspect.signature(AccFG.run))

print("\nAccFG.run_mol signature:")
# Assuming run_mol is a method
if hasattr(AccFG, 'run_mol'):
    print(inspect.signature(AccFG.run_mol))
else:
    print("AccFG has no attribute run_mol")
    
# Check source code if possible via inspect
try:
    print("\nSource of AccFG.run_mol:")
    print(inspect.getsource(AccFG.run_mol))
except Exception as e:
    print(e)
