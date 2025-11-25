import os
import sys
import logging
from typing import Dict, Any

import runpod

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pow.compute.compute import Compute
from pow.models.utils import Params
from pow.random import get_target

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global compute instance (loaded once, reused across requests)
COMPUTE = None
CURRENT_BLOCK_HASH = None


def initialize_compute(
    block_hash: str,
    block_height: int,
    params_dict: Dict[str, Any] = None,
    devices: list = None,
) -> Compute:
    """Initialize the compute model for a specific block hash"""
    global COMPUTE, CURRENT_BLOCK_HASH

    # If compute already initialized for this block_hash, reuse it
    if COMPUTE is not None and CURRENT_BLOCK_HASH == block_hash:
        logger.info(f"Reusing existing compute for block_hash={block_hash}")
        return COMPUTE

    logger.info(f"Initializing compute for block_hash={block_hash}")

    # Default params if not provided
    if params_dict is None:
        params_dict = {}

    params = Params(**params_dict)

    # Default devices if not provided
    if devices is None:
        devices = ["cuda:0"]

    # Create compute instance
    COMPUTE = Compute(
        params=params,
        block_hash=block_hash,
        block_height=block_height,
        public_key="",  # Will be overridden per request
        r_target=0.0,   # Will be overridden per request
        devices=devices,
        node_id=0,      # Serverless doesn't need node_id
    )

    CURRENT_BLOCK_HASH = block_hash
    logger.info("Compute initialized successfully")

    return COMPUTE


def handler(event: Dict[str, Any]):
    """
    Runpod handler function - supports both batch and streaming modes

    Batch mode (legacy) - provide "nonces" array:
    {
        "block_hash": "string",
        "block_height": int,
        "public_key": "string",
        "nonces": [int, int, ...],
        "r_target": float,
        "params": {...},  # optional
        "devices": ["cuda:0", ...]  # optional
    }

    Streaming mode - provide "streaming" flag:
    {
        "block_hash": "string",
        "block_height": int,
        "public_key": "string",
        "r_target": float,
        "streaming": true,
        "batch_size": 16,  # optional, default 16
        "max_batches": 100,  # optional, default unlimited
        "start_nonce": 0,  # optional, default 0
        "params": {...},
        "devices": ["cuda:0", ...]
    }

    Returns (batch mode):
    {
        "public_key": "string",
        "block_hash": "string",
        "block_height": int,
        "nonces": [int, ...],
        "dist": [float, ...],
        "node_id": int,
        "total_computed": int,
        "total_valid": int
    }

    Yields (streaming mode):
    Multiple batches of results as they are computed
    """
    try:
        # Extract input data
        input_data = event.get("input", {})

        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        r_target = input_data["r_target"]
        params_dict = input_data.get("params", {})
        devices = input_data.get("devices", ["cuda:0"])

        # Initialize compute (or reuse existing)
        compute = initialize_compute(
            block_hash=block_hash,
            block_height=block_height,
            params_dict=params_dict,
            devices=devices,
        )

        # Get target for this block
        target = get_target(block_hash, compute.params.vocab_size)

        # Check if streaming mode or batch mode
        if input_data.get("streaming", False):
            # STREAMING MODE
            batch_size = input_data.get("batch_size", 16)
            max_batches = input_data.get("max_batches", None)  # None = unlimited
            start_nonce = input_data.get("start_nonce", 0)

            logger.info(f"Streaming mode: block_hash={block_hash}, public_key={public_key[:10]}..., "
                       f"batch_size={batch_size}, max_batches={max_batches}, r_target={r_target}")

            current_nonce = start_nonce
            batch_count = 0
            total_computed = 0
            total_valid = 0

            # Generator loop - yield results as batches are computed
            while True:
                # Check if we've reached max_batches
                if max_batches is not None and batch_count >= max_batches:
                    logger.info(f"Reached max_batches={max_batches}, stopping")
                    break

                # Generate batch of nonces
                nonces = list(range(current_nonce, current_nonce + batch_size))

                # Compute distances for this batch
                proof_batch = compute(
                    nonces=nonces,
                    public_key=public_key,
                    target=target,
                )

                # Filter by r_target
                filtered_batch = proof_batch.sub_batch(r_target)

                batch_count += 1
                total_computed += len(proof_batch)
                total_valid += len(filtered_batch)

                logger.info(f"Batch {batch_count}: computed {len(proof_batch)} nonces, "
                           f"{len(filtered_batch)} passed filter")

                # Yield this batch result
                yield {
                    "public_key": filtered_batch.public_key,
                    "block_hash": filtered_batch.block_hash,
                    "block_height": filtered_batch.block_height,
                    "nonces": filtered_batch.nonces,
                    "dist": filtered_batch.dist,
                    "node_id": filtered_batch.node_id,
                    "batch_number": batch_count,
                    "batch_computed": len(proof_batch),
                    "batch_valid": len(filtered_batch),
                    "total_computed": total_computed,
                    "total_valid": total_valid,
                    "next_nonce": current_nonce + batch_size,
                }

                # Move to next batch
                current_nonce += batch_size

        else:
            # BATCH MODE (legacy)
            nonces = input_data["nonces"]

            logger.info(f"Batch mode: block_hash={block_hash}, public_key={public_key[:10]}..., "
                       f"nonces={len(nonces)}, r_target={r_target}")

            # Compute distances
            proof_batch = compute(
                nonces=nonces,
                public_key=public_key,
                target=target,
            )

            # Filter by r_target
            filtered_batch = proof_batch.sub_batch(r_target)

            logger.info(f"Computed {len(proof_batch)} nonces, {len(filtered_batch)} passed r_target filter")

            # Return single result
            return {
                "public_key": filtered_batch.public_key,
                "block_hash": filtered_batch.block_hash,
                "block_height": filtered_batch.block_height,
                "nonces": filtered_batch.nonces,
                "dist": filtered_batch.dist,
                "node_id": filtered_batch.node_id,
                "total_computed": len(proof_batch),
                "total_valid": len(filtered_batch),
            }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Start the serverless handler
runpod.serverless.start({"handler": handler})
