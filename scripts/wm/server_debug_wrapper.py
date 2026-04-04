"""Wrapper that injects debugpy into rank 0 before running the server."""
import os
import logging

rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
debug_port = int(os.environ.get("DEBUG_PORT", "5678"))

if rank == 0 and os.environ.get("ENABLE_DEBUGPY", ""):
    import debugpy
    debugpy.listen(("0.0.0.0", debug_port))
    print(f"[debugpy] Rank 0 listening on 0.0.0.0:{debug_port}, waiting for VSCode to attach...")
    debugpy.wait_for_client()
    print("[debugpy] Debugger attached!")

# Run the original entry point
logging.basicConfig(level=logging.INFO, force=True)

import tyro
from socket_test_optimized_AR import Args, main

args = tyro.cli(Args)
main(args)
