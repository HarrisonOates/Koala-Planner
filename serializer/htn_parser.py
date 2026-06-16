import sys
import json
from htn_serializer import HTNSerializer
from fond_merger import FONDMerger

if __name__ == "__main__" :
    serializer = HTNSerializer(sys.argv[1])
    result = serializer.run()
    merger = FONDMerger(result)
    result = merger.run()

    # Load probability map if provided as third argument
    prob_map = None
    if len(sys.argv) > 3:
        prob_map_path = sys.argv[3]
        with open(prob_map_path, 'r') as f:
            prob_map = json.load(f)

    merger.inject_probabilities(prob_map)

    with open(sys.argv[2], "w+") as f:
        json.dump(result, f, indent=4)
