import multiprocessing as mp
import time
import sys

# Import the nodes
from vision.vision_node import run_vision_node
# We will create this file next
from BiLSTM.inference_node import run_inference_node 

def main():
    # Enforce 'spawn' for safe CUDA initialization in Process 2
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Already set

    print("[Main] Booting Driver Monitoring System...")

    # Initialize the GPU Inference Process (Process 2)
    p_inference = mp.Process(target=run_inference_node, name="InferenceNode")
    
    # Start the GPU node first so the ZMQ PULL/PUB sockets are ready
    p_inference.start()
    
    # Give the GPU process a second to initialize PyTorch and bind sockets
    print("[Main] Waiting for Inference Node to initialize...")
    time.sleep(4)
    if not p_inference.is_alive():
        print("[Main] Inference Node failed to start. Exiting.")
        sys.exit(1)

    try:
        # Run the Vision Node (Process 1) on the main thread.
        # OpenCV's cv2.imshow requires being on the main thread on some OS (macOS/Windows)
        print("[Main] Launching Vision Node on the main thread...")
        run_vision_node()
        
    except KeyboardInterrupt:
        print("\n[Main] Shutdown signal received.")
    except Exception as e:
        print(f"[Main] Exception occurred: {e}")
    finally:
        # Clean up the GPU process when the Vision Node exits (e.g., user presses 'q')
        print("[Main] Terminating Inference Node...")
        p_inference.terminate()
        p_inference.join()
        print("[Main] System shutdown complete.")

if __name__ == '__main__':
    main()