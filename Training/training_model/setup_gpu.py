import os
import subprocess
import sys

# Modify the setup_gpu_environment function
def setup_gpu_environment():
    """Set up the environment for GPU usage with TensorFlow"""
    print("Setting up GPU environment for TensorFlow...")
    
    # Install the correct TensorFlow version for better multiprocessing support
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.15.0", "--force-reinstall"])
    
    # Set environment variables
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Improve GPU thread handling
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Use async memory allocator
    
    # Check CUDA installation
    try:
        import tensorflow as tf
        print("\nTensorFlow version:", tf.__version__)
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s): {gpus}")
            
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu}")
            
            # Test GPU with a simple operation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"Matrix multiplication result: {c}")
                print(f"Tensor device: {c.device}")
                
            print("\nGPU setup successful!")
        else:
            print("No GPU devices found by TensorFlow.")
            print("Checking CUDA installation...")
            print(f"CUDA available: {tf.test.is_built_with_cuda()}")
            print(f"GPU available: {tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else 'Unknown'}")
            
            # Additional diagnostics
            print("\nRunning nvidia-smi:")
            subprocess.run(["nvidia-smi"])
            
            print("\nChecking CUDA libraries:")
            cuda_libs = [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/lib64/libcublas.so",
                "/usr/local/cuda/lib64/libcufft.so",
                "/usr/local/cuda/lib64/libcurand.so",
                "/usr/local/cuda/lib64/libcusolver.so",
                "/usr/local/cuda/lib64/libcusparse.so"
            ]
            
            for lib in cuda_libs:
                if os.path.exists(lib):
                    print(f"✓ {lib} exists")
                else:
                    print(f"✗ {lib} missing")
            
            print("\nPlease install the required CUDA libraries for TensorFlow.")
    except Exception as e:
        print(f"Error during GPU setup: {e}")

if __name__ == "__main__":
    setup_gpu_environment()