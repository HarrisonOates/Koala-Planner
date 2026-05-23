import sys
import os
import subprocess
import json
import resource

# Timeout in minutes
def solve(domain, problem, optional_flags, timeout=30, mem_limit_gb=None):
    path = os.getcwd()
    parser_path = path + "/parser/pandaPIparser"
    grounder_path = path + "/grounder/pandaPIgrounder/"
    serilazer_path = path + "/serializer/"
    planner_path = path + "/planner/"

    # Check for probabilistic and preprocess if needed
    sys.path.insert(0, serilazer_path)
    from prob_preprocessor import has_probabilistic, preprocess

    domain_path = os.path.abspath(domain)
    prob_map = None
    prob_map_path = None
    cleaned_domain = None

    if has_probabilistic(domain_path):
        print("Detected probabilistic in domain, preprocessing...")
        cleaned_domain, prob_map = preprocess(domain_path)
        domain_for_parser = cleaned_domain
        # Write prob_map to a temp file for the serializer
        prob_map_path = os.path.join(serilazer_path, "prob_map.json")
        with open(prob_map_path, 'w') as f:
            json.dump(prob_map, f)
        print(f"Probability map: {prob_map}")
    else:
        domain_for_parser = domain_path

    # Parsing
    parsed = subprocess.run(
        [parser_path,
         domain_for_parser, os.path.abspath(problem)],
         capture_output=True)
    print(parsed.stderr.decode("utf-8"))
    with open(grounder_path + "parsed.htn", "w+") as f:
        f.write(parsed.stdout.decode("utf-8"))

    # Clean up temp domain file
    if cleaned_domain and os.path.isfile(cleaned_domain):
        os.remove(cleaned_domain)

    # Grounding
    if os.path.isfile(grounder_path + "parsed.htn"):
        subprocess.run(
            [grounder_path + "pandaPIgrounder",
            grounder_path + "parsed.htn",
            serilazer_path + "result.sas+"], capture_output=True
        )
        os.remove(grounder_path + "parsed.htn")
    else:
        print(f"\t\tfailed to parse {problem}", file=sys.stderr)
        return
    # Serializing
    if os.path.isfile(serilazer_path + "result.sas+"):
        serialize_cmd = [
            "python3", serilazer_path + "htn_parser.py",
            serilazer_path + "result.sas+", planner_path + "result.json"
        ]
        if prob_map_path and os.path.isfile(prob_map_path):
            serialize_cmd.append(prob_map_path)
        serialized = subprocess.run(serialize_cmd, capture_output=True)
        print(serialized.stderr.decode("utf-8"))
        os.remove(serilazer_path + "result.sas+")
        if prob_map_path and os.path.isfile(prob_map_path):
            os.remove(prob_map_path)
    else:
        print(f"\t\tfailed to ground {problem}", file=sys.stderr)
        return
    # Search
    if os.path.isfile(planner_path + "result.json"):
        try:
            release_suffix = "target/release/planner"
            debug_suffix = "target/debug/planner"
            debug_mode = False

            mem_preexec = None
            if mem_limit_gb is not None:
                limit_bytes = int(mem_limit_gb * 1024 ** 3)
                print(f"Memory limit: {mem_limit_gb} GB ({limit_bytes} bytes)")
                mem_preexec = lambda: resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

            if os.path.isfile(planner_path + release_suffix):
                print("Running in release mode")
                result = subprocess.run(
                    [planner_path + release_suffix, planner_path + "result.json"] + optional_flags,
                    stdout=subprocess.PIPE, stderr=None,
                    timeout=60 * timeout, preexec_fn=mem_preexec)
            elif os.path.isfile(planner_path + debug_suffix):
                print("Release binary not available, using debug binary...")
                debug_mode = True
                result = subprocess.run(
                    [planner_path + debug_suffix, planner_path + "result.json"] + optional_flags,
                    capture_output=False, timeout=60 * timeout, preexec_fn=mem_preexec)
            else:
                print(f"No binary found in {planner_path + release_suffix} or {planner_path + debug_suffix}, exiting.")
                sys.exit(1)
            if not debug_mode:
                with open(path + f"/{problem}_solution_{''.join(optional_flags)}.txt", "x") as f:
                    f.write(result.stdout.decode("utf-8"))
        except subprocess.TimeoutExpired:
            print(f'\t\ttimeout for {problem}')
        try:
            os.remove(planner_path + "result.json")
        except FileNotFoundError:
            pass
    else:
        print(f"failed to serialize {problem}", file=sys.stderr)

if __name__ == "__main__":
    import sys
    domain = sys.argv[1]
    problem = sys.argv[2]
    args = sys.argv[3:]

    # Parse named flags from remaining arguments
    mode_flag = []
    heuristic_flag = []
    threshold_flag = []
    tiebreak_flag = []
    mem_limit_gb = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--fixed", "--flexible", "--andstar"):
            mode_flag = [arg]
        elif arg in ("--ff", "--add", "--max", "--prob"):
            heuristic_flag = [arg]
        elif arg == "--threshold" and i + 1 < len(args):
            threshold_flag = ["--threshold", args[i + 1]]
            i += 1
        elif arg == "--tiebreak" and i + 1 < len(args):
            tiebreak_flag = ["--tiebreak", args[i + 1]]
            i += 1
        elif arg == "--mem-limit" and i + 1 < len(args):
            mem_limit_gb = float(args[i + 1])
            i += 1
        i += 1

    # Default heuristic to --ff if not specified
    if not heuristic_flag:
        print("No heuristic specified, defaulting to --ff")
        heuristic_flag = ["--ff"]

    optional_flags = mode_flag + heuristic_flag + threshold_flag + tiebreak_flag
    solve(domain, problem, optional_flags, mem_limit_gb=mem_limit_gb)
