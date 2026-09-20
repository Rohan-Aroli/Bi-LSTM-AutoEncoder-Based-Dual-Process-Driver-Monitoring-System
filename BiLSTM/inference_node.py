import zmq
import torch
import torch.nn as nn
import numpy as np
import config
from BiLSTM.Model import BiLSTMAutoencoder
from BiLSTM.mse_calibrator import MSEBaselineManager
# 🚨 YOU MUST TUNE THIS VALUE 🚨
# If the reconstruction error is higher than this, it's an anomaly.

mse_mgr= MSEBaselineManager()

# ANOMALY_THRESHOLD = 0.5 

def run_inference_node():
    print("[Inference Node] Booting up Autoencoder server...")
    context = zmq.Context()

    pull_sock = context.socket(zmq.PULL)
    pull_sock.bind(f"tcp://127.0.0.1:{config.ZMQ_PUSH_PORT}")

    pub_sock = context.socket(zmq.PUB)
    pub_sock.bind(f"tcp://127.0.0.1:{config.ZMQ_SUB_PORT}")

    # Hardware acceleration setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAutoencoder(hidden_dim=32).to(device)
    
    # Load your pre-trained weights
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        print(f"[Inference Node] Weights loaded successfully on {device}")
    except Exception as e:
        print(f"[Inference Node] CRITICAL ERROR loading weights: {e}")
        return

    model.eval() # Lock network for inference
    
    # We use Mean Squared Error to measure how badly the model failed to rebuild the input
    criterion = nn.MSELoss() 

    while True:
       
        try:
            payload = pull_sock.recv_pyobj()
            payload_type = payload.get("type", "INFERENCE")
            buffer_data = payload["buffer"]
            
            # --- THE NEW CALIBRATION CATCHER ---
            if payload_type == "CALIBRATE_MSE":
                print(f"[Inference Node] Received {len(buffer_data)} frames for MSE Calibration.")
                # Your compute_and_save method handles the 60-frame overlapping slices natively
                ANOMALY_THRESHOLD = mse_mgr.compute_and_save(model, device, buffer_data)
                if ANOMALY_THRESHOLD is None:
                    Anamaly_Threshold = 0.5 # Fallback threshold if calibration fails
                    print("[Inference Node] WARNING:MSE Calibration failed. Using default threshold of 0.5.")
                # Tell the vision node we are done
                pub_sock.send_pyobj({"state": "NORMAL"}) 
                continue
            # 2. Convert to Tensor: (60, 8) -> (1, 60, 8)
            input_tensor = torch.tensor(buffer_data, dtype=torch.float32).unsqueeze(0).to(device)

            # 3. Execute the forward pass and measure the error
            with torch.no_grad():
                reconstructed_tensor = model(input_tensor)
                
                # Calculate the MSE loss between the real input and the AI's attempt to rebuild it
                reconstruction_error = criterion(reconstructed_tensor, input_tensor).item()

            # 4. The Anomaly Gate
            # High error = The sequence looks nothing like normal driving = Drowsy/Distracted
            if reconstruction_error > ANOMALY_THRESHOLD:
                state = "DROWSY"
            else:
                state = "NORMAL"

            # (Optional) Print the error to the console to help you tune the threshold
            # print(f"Error: {reconstruction_error:.4f} | State: {state}")

            # 5. Broadcast the result
            pub_sock.send_pyobj({"state": state})

        except KeyboardInterrupt:
            print("\n[Inference Node] Shutting down server...")
            break
        except Exception as e:
            pass

    pull_sock.close()
    pub_sock.close()
    context.term()