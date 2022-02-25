from deep_morpho.morp_operations import ParallelMorpOperations


morp_operations = []

for op in [
    "disk",
    "hstick",
    "dcross",
]:
    size = 7
    if op == "disk":
        size = size // 2

    morp_operations.append(ParallelMorpOperations.dilation((op, size)))
    morp_operations.append(ParallelMorpOperations.erosion((op, size)))
    morp_operations.append(ParallelMorpOperations.closing((op, size)))
    morp_operations.append(ParallelMorpOperations.opening((op, size)))
